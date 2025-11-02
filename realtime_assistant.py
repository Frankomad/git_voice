#!/usr/bin/env python3
"""
Simple AI Voice Assistant using OpenAI Realtime API
Runs in terminal with microphone input and speaker output
"""

import asyncio
import json
import os
import sys
from typing import Optional
import websockets
from dotenv import load_dotenv

# Try to import pyaudio, fallback if not available
try:
    import pyaudio
    PYTHON_AUDIO_AVAILABLE = True
except ImportError:
    PYTHON_AUDIO_AVAILABLE = False
    print("Warning: pyaudio not installed. Audio will not work.")
    print("Install it with: pip install pyaudio")

load_dotenv()

# Audio configuration
SAMPLE_RATE = 24000  # Realtime API expects 24kHz
CHUNK_SIZE = 480  # 20ms chunks at 24kHz
AUDIO_FORMAT = pyaudio.paInt16 if PYTHON_AUDIO_AVAILABLE else None
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit

# Realtime API endpoint
REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"


class RealtimeAssistant:
    def __init__(self, api_key: str, qa_file: str = "mooveon_qa.json", speech_rate: float = None, voice: str = None):
        self.api_key = api_key
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_input_stream: Optional[object] = None
        self.audio_output_stream: Optional[object] = None
        self.is_running = False
        self.session_configured = False
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.qa_data = self.load_qa_data(qa_file)
        
        # Speech rate configuration (0.25 to 4.0, default 1.0)
        # Can be set via environment variable or parameter
        if speech_rate is None:
            speech_rate = float(os.getenv("SPEECH_RATE", "1.0"))
        self.speech_rate = max(0.25, min(4.0, speech_rate))  # Clamp between 0.25 and 4.0
        
        # Voice configuration
        # Available voices: cedar, marin, echo, breeze, nova, ember, sky, orion, alloy, fable, onyx, shimmer
        # Can be set via environment variable or parameter
        if voice is None:
            voice = os.getenv("VOICE", "alloy")
        self.voice = voice.lower()  # Normalize to lowercase
        
        if PYTHON_AUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None
    
    def load_qa_data(self, path: str) -> dict:
        """Load Q&A data from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Q&A file {path} not found. Using default.")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {path}. Using default.")
            return {}
    
    def build_system_prompt(self) -> str:
        """Build minimal system prompt focused ONLY on Croatian language"""
        
        qa = self.qa_data
        company_name = qa.get('company', 'Moove On')
        greeting = qa.get('greeting', 'Dobar dan, dobili ste MooveOn asistenta.')
        description = qa.get('description', '')
        faq_list = qa.get('faq', [])
        key_info = qa.get('key_information', {})
        
        # Build FAQ section
        faq_section = ""
        if faq_list:
            faq_section = "\n\nFREQUENTLY ASKED QUESTIONS:\n"
            for item in faq_list:
                faq_section += f"- Q: {item.get('pitanje', '')}\n"
                faq_section += f"  A: {item.get('odgovor', '')}\n"
        
        # Build key information section
        key_info_section = ""
        if key_info:
            key_info_section = "\n\nKEY INFORMATION:\n"
            for key, value in key_info.items():
                key_info_section += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        # Select general questions to suggest when user asks unrelated questions
        general_questions = []
        if faq_list:
            # Find some general/interesting questions
            for item in faq_list:
                q = item.get('pitanje', '')
                # Select general questions that might interest most users
                if any(keyword in q.lower() for keyword in ['čime se bavi', 'kako ugovorit', 'tko sve može', 'koje marke', 'što je moove']):
                    general_questions.append(q)
                # Limit to 3-4 general questions
                if len(general_questions) >= 4:
                    break
        
        # If no specific general questions found, use first few
        if not general_questions and faq_list:
            general_questions = [item.get('pitanje', '') for item in faq_list[:3]]
        
        general_questions_text = ""
        if general_questions:
            general_questions_text = "\n\nGENERAL QUESTIONS TO SUGGEST (use when user asks unrelated questions):\n"
            for i, q in enumerate(general_questions, 1):
                general_questions_text += f"- {q}\n"
        
        # Ultra minimal prompt - Written in English so model clearly understands
        system_prompt = f"""You are a {company_name} assistant. You MUST speak ONLY in Croatian language (Hrvatski jezik). Croatian is your ONLY allowed language.

COMPANY INFORMATION:
- Company: {company_name}
- Description: {description}

CRITICAL LANGUAGE RULES:
- You are FORBIDDEN to speak in any other language. You MUST speak ONLY Croatian.
- If the user speaks in another language (Portuguese, Spanish, English, Serbian, Bosnian, or any other language), you MUST respond with this exact phrase in Croatian: "Oprostite, pričam samo na hrvatskom jeziku."
- Every single word you speak MUST be in Croatian language. Use Croatian ijekavica variant (NOT ekavica, NOT Serbian, NOT Bosnian).
- Before responding, verify: "Am I speaking Croatian?" If the answer is no, do not respond.

FIRST CONTACT:
- When the conversation starts, you MUST introduce yourself with this greeting: "{greeting}"
- Always greet the user first with this exact phrase or similar variations in Croatian.

{key_info_section}

{faq_section}

{general_questions_text}

HOW TO RESPOND:
- When user asks a question, find the matching Q&A from the FAQ list above and provide the answer in Croatian.
- If you find a similar question in FAQ, use that answer.
- Always respond in Croatian language only.
- Provide accurate information based on the FAQ above.

CRITICAL: HANDLING UNRELATED QUESTIONS:
- If the user asks a question that is NOT related to {company_name}, Moove On services, vehicle rental, leasing, or any topic in the FAQ list above, you MUST respond with:
  "Nažalost vam ne mogu dati odgovor na to pitanje. Zanima li vas možda [suggest one general question from the GENERAL QUESTIONS list above]?"
- Always suggest ONE relevant general question from the GENERAL QUESTIONS list when responding to unrelated questions.
- The suggested question should be relevant and interesting for most users.
- Use your judgment to pick the most appropriate general question based on the context.

Example responses in Croatian:
- First greeting: "{greeting}"
- If asked about the company: "{faq_list[0].get('odgovor', '') if faq_list else description}"
- If asked how to request an offer: Use the answer from FAQ about requesting an offer.
- If asked unrelated question (e.g., "What's the weather?"): "Nažalost vam ne mogu dati odgovor na to pitanje. Zanima li vas možda {general_questions[0] if general_questions else 'Čime se bavi Moove On?'}?"

Remember: Your ONLY language is Croatian. No exceptions."""
        
        return system_prompt
    
    async def connect(self):
        """Connect to OpenAI Realtime API via WebSocket"""
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1")
        ]
        
        print("Connecting to OpenAI Realtime API...")
        self.websocket = await websockets.connect(
            REALTIME_API_URL,
            additional_headers=headers,
            ping_interval=20,  # Send ping every 20 seconds to keep connection alive
            ping_timeout=10,   # Wait 10 seconds for pong response
            close_timeout=10   # Wait 10 seconds before closing
        )
        print("Connected!")
    
    def setup_audio(self):
        """Setup audio input and output streams"""
        if not PYTHON_AUDIO_AVAILABLE:
            print("Audio not available. Please install pyaudio.")
            return False
        
        try:
            # Open separate input stream
            self.audio_input_stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self.audio_callback
            )
            self.audio_input_stream.start_stream()
            
            # Open separate output stream
            def output_callback(in_data, frame_count, time_info, status):
                # This will be fed from audio_output_queue
                return (None, pyaudio.paContinue)
            
            self.audio_output_stream = self.audio.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            self.audio_output_stream.start_stream()
            
            print("Audio streams ready")
            return True
        except Exception as e:
            print(f"Error setting up audio: {e}")
            return False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input - store in queue"""
        if self.is_running:
            try:
                self.audio_input_queue.put_nowait(in_data)
            except Exception:
                # Queue operation failed, skip this chunk
                pass
        
        # Output will be handled by separate audio output stream
        return (None, pyaudio.paContinue)
    
    async def audio_input_sender(self):
        """Continuously send audio input to Realtime API"""
        import base64
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Get audio chunk from queue with timeout
                audio_data = await asyncio.wait_for(
                    self.audio_input_queue.get(),
                    timeout=0.1
                )
                
                if self.websocket and self.is_running:
                    # Send audio as base64 encoded PCM16
                    # We'll catch exceptions if connection is closed rather than checking
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    
                    event = {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64
                    }
                    await self.websocket.send(json.dumps(event))
                    consecutive_errors = 0  # Reset error counter on success
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("\n>>> WebSocket connection closed in audio sender. Attempting to reconnect...")
                await self.reconnect()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if self.is_running:
                    if "keepalive" in str(e).lower() or "ping" in str(e).lower():
                        # Keepalive timeout - try to reconnect
                        print(f"\n>>> Keepalive timeout detected. Attempting to reconnect... (error {consecutive_errors}/{max_consecutive_errors})")
                        if consecutive_errors >= max_consecutive_errors:
                            await self.reconnect()
                            consecutive_errors = 0
                    else:
                        print(f"Error sending audio: {e} (error {consecutive_errors}/{max_consecutive_errors})")
                
                # If too many consecutive errors, try to reconnect
                if consecutive_errors >= max_consecutive_errors:
                    print("\n>>> Too many consecutive errors. Attempting to reconnect...")
                    await self.reconnect()
                    consecutive_errors = 0
                    
                # Small delay before retrying to avoid spamming errors
                await asyncio.sleep(0.1)
    
    async def reconnect(self):
        """Reconnect to OpenAI Realtime API and reconfigure session"""
        try:
            # Close existing connection if any
            if self.websocket:
                try:
                    # Check if websocket is closed (handle different websocket versions)
                    is_closed = getattr(self.websocket, 'closed', True)
                    if not is_closed:
                        await self.websocket.close()
                except (AttributeError, TypeError, Exception):
                    # If we can't check or close, just continue
                    pass
            
            # Reset session configuration flag
            self.session_configured = False
            
            # Wait a moment before reconnecting
            await asyncio.sleep(1)
            
            # Reconnect
            await self.connect()
            print("Reconnected successfully!")
            
            # Wait for session.created event and reconfigure
            # The handle_responses will handle session configuration automatically
            
        except Exception as e:
            print(f"Error reconnecting: {e}")
            await asyncio.sleep(2)  # Wait before retrying
    
    async def configure_session(self):
        """Send session configuration after session is created"""
        # Build system prompt from mooveon Q&A data
        system_instructions = self.build_system_prompt()
        
        # Debug: Print system prompt
        print(f"[DEBUG] System prompt length: {len(system_instructions)} characters")
        print(f"[DEBUG] System prompt:\n{system_instructions}")
        
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": system_instructions,
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 500,
                    "silence_duration_ms": 2000
                },
                "output_audio_format": "pcm16",
                "voice": self.voice,
                "temperature": 0.7,
                "max_response_output_tokens": 4096,
                "speed": self.speech_rate  # Speech rate: 0.25 (slow) to 4.0 (fast), default 1.0
            }
        }
        
        await self.websocket.send(json.dumps(session_update))
        print("Session configured")
        print(f"[DEBUG] Sent session update with instructions ({len(system_instructions)} chars)")
        print(f"[DEBUG] Speech rate: {self.speech_rate}x (0.25 = slow, 1.0 = normal, 4.0 = fast)")
        print(f"[DEBUG] Voice: {self.voice}")
    
    async def handle_responses(self):
        """Handle responses from Realtime API"""
        audio_buffer = bytearray()
        
        try:
            async for message in self.websocket:
                if not self.is_running:
                    break
                
                data = json.loads(message)
                event_type = data.get("type")
                
                # Debug: Print all events to see what's happening
                if event_type not in ["response.audio_transcript.delta"]:  # Skip delta spam
                    if event_type in ["conversation.item.input_audio_transcription.completed", 
                                     "input_audio_buffer.speech_started",
                                     "input_audio_buffer.speech_stopped",
                                     "response.output_item.created"]:
                        print(f"[DEBUG] Full data: {json.dumps(data, indent=2, ensure_ascii=False)}")
                
                if event_type == "session.created":
                    print("Session created successfully")
                    if not self.session_configured:
                        await self.configure_session()
                        self.session_configured = True
                
                elif event_type == "session.updated":
                    print("Session updated")
                
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcription = data.get("transcript", "")
                    if transcription:
                        print(f"\n>>> USER SAID: {transcription}")
                    else:
                        print(f"\n>>> RECEIVED TRANSCRIPTION EVENT BUT NO TRANSCRIPT (might be empty/background noise)")
                
                elif event_type == "conversation.item.input_audio_transcription.failed":
                    error = data.get("error", {})
                    print(f"\n>>> TRANSCRIPTION FAILED: {error}")
                
                elif event_type == "input_audio_buffer.speech_started":
                    print(f"\n>>> SPEECH DETECTED - User started speaking")
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print(f"\n>>> SPEECH STOPPED - User finished speaking")
                
                elif event_type == "response.audio_transcript.created":
                    print(f"\n>>> ASSISTANT STARTING RESPONSE...")
                
                elif event_type == "response.output_item.created":
                    item_type = data.get("item", {}).get("type", "unknown")
                    print(f"\n>>> ASSISTANT CREATED OUTPUT ITEM: {item_type}")
                
                elif event_type == "response.audio_transcript.delta":
                    transcript_delta = data.get("delta", "")
                    if transcript_delta:
                        sys.stdout.write(transcript_delta)
                        sys.stdout.flush()
                
                elif event_type == "response.audio_transcript.done":
                    transcript = data.get("transcript", "")
                    if transcript:
                        print(f"\n[Assistant]: {transcript}\n")
                
                elif event_type == "response.audio.delta":
                    # Accumulate audio chunks
                    audio_chunk = data.get("delta", "")
                    if audio_chunk:
                        import base64
                        try:
                            audio_data = base64.b64decode(audio_chunk)
                            audio_buffer.extend(audio_data)
                        except Exception as e:
                            print(f"Error decoding audio: {e}")
                
                elif event_type == "response.audio.done":
                    # Play accumulated audio
                    if audio_buffer:
                        self.play_audio(bytes(audio_buffer))
                        audio_buffer.clear()
                
                elif event_type == "response.output_item.done":
                    # Response item completed
                    pass
                
                elif event_type == "error":
                    error = data.get("error", {})
                    print(f"\n>>> ERROR: {error.get('message', 'Unknown error')}")
                    print(f"[DEBUG] Error details: {json.dumps(error, indent=2, ensure_ascii=False)}")
                
                elif event_type == "response.done":
                    print("\n>>> RESPONSE DONE - Assistant finished responding")
                
                else:
                    # Print any unhandled events for debugging
                    if event_type not in ["response.audio_transcript.delta", "response.audio.delta"]:
                        print(f"[DEBUG] Unhandled event: {event_type}")
        
        except websockets.exceptions.ConnectionClosed:
            print("\n>>> Connection closed in response handler")
            if self.is_running:
                print("Attempting to reconnect...")
                await self.reconnect()
        except Exception as e:
            print(f"\nError in response handler: {e}")
            if self.is_running and ("keepalive" in str(e).lower() or "ping" in str(e).lower()):
                print("Attempting to reconnect due to keepalive timeout...")
                await self.reconnect()
    
    def play_audio(self, audio_data: bytes):
        """Play audio output"""
        if self.audio_output_stream and PYTHON_AUDIO_AVAILABLE:
            try:
                # Write audio to output stream
                self.audio_output_stream.write(audio_data, exception_on_underflow=False)
            except Exception as e:
                print(f"Error playing audio: {e}")
    
    async def run(self):
        """Main run loop"""
        try:
            await self.connect()
            
            if not self.setup_audio():
                print("Failed to setup audio. Exiting.")
                return
            
            self.is_running = True
            self.loop = asyncio.get_event_loop()
            
            print("\n" + "="*50)
            print("Voice Assistant Ready!")
            print("Speak into your microphone...")
            print("Press Ctrl+C to exit")
            print("="*50 + "\n")
            
            # Start response handler and audio input sender concurrently
            await asyncio.gather(
                self.handle_responses(),
                self.audio_input_sender()
            )
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.audio_input_stream:
            self.audio_input_stream.stop_stream()
            self.audio_input_stream.close()
        
        if self.audio_output_stream:
            self.audio_output_stream.stop_stream()
            self.audio_output_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        if self.websocket:
            await self.websocket.close()
        
        print("Cleaned up")


async def main():
    """Main entry point"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your API key:")
        print("OPENAI_API_KEY=your-api-key-here")
        return
    
    # Optional: Configure speech rate and voice
    # Option 1: Use environment variables (SPEECH_RATE and VOICE in .env file)
    # Option 2: Set them directly in code (see examples below):
    
    assistant = RealtimeAssistant(api_key)  # Uses .env settings or defaults
    
    # Examples with custom settings:
    # assistant = RealtimeAssistant(api_key, speech_rate=1.5, voice="nova")  # Fast speech with Nova voice
    # assistant = RealtimeAssistant(api_key, voice="cedar")  # Change voice only
    # assistant = RealtimeAssistant(api_key, speech_rate=0.75)  # Slow speech only
    
    await assistant.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

