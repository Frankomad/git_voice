#!/usr/bin/env python3
"""
Twilio Webhook Server for AI Agent
Handles incoming SIP calls from Twilio and bridges them to OpenAI Realtime API
"""

# Monkey patch must be done before importing any other modules
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, Response
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse
import os
import asyncio
import json
import websockets
import base64
import struct
import numpy as np
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv
from realtime_assistant import RealtimeAssistant

load_dotenv()

app = Flask(__name__)
sock = Sock(app)  # Enable WebSocket support in Flask

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REALTIME_API_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
NGROK_URL = os.getenv("NGROK_URL", "")  # Legacy: ngrok URL (if using ngrok)
BASE_URL = os.getenv("BASE_URL") or os.getenv("RENDER_URL") or NGROK_URL  # Base URL for service (Render or ngrok)

# Audio conversion constants
TWILIO_SAMPLE_RATE = 8000  # Twilio uses 8kHz μ-law
OPENAI_SAMPLE_RATE = 24000  # OpenAI Realtime API uses 24kHz PCM16

# Build μ-law to PCM16 lookup table once (ITU-T G.711 standard)
_ULAW_TO_PCM_TABLE = None

def _build_ulaw_table():
    """Build μ-law to PCM16 lookup table (called once)"""
    global _ULAW_TO_PCM_TABLE
    if _ULAW_TO_PCM_TABLE is not None:
        return _ULAW_TO_PCM_TABLE
    
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        # Complement all bits (μ-law encoding complements bits)
        complemented = (~i) & 0xFF
        
        # Extract sign bit (bit 7 after complement: 0 = positive, 1 = negative)
        sign_bit = (complemented >> 7) & 0x01
        
        # Extract exponent (bits 6-4 after complement)
        exponent = (complemented >> 4) & 0x07
        
        # Extract mantissa (bits 3-0 after complement)
        mantissa = complemented & 0x0F
        
        # Calculate PCM value using ITU-T G.711 formula
        # PCM = sign * (((mantissa << 1) + 33) << exponent) - 33)
        pcm = (((mantissa << 1) + 33) << exponent) - 33
        
        # Apply sign (after complement, sign_bit 1 means negative)
        if sign_bit == 1:
            pcm = -pcm
        
        # Clip to int16 range
        pcm = max(-32768, min(32767, pcm))
        
        table[i] = np.int16(pcm)
    
    _ULAW_TO_PCM_TABLE = table
    return table

def ulaw_to_pcm16(ulaw_audio: bytes) -> bytes:
    """Convert μ-law audio to 16-bit PCM using ITU-T G.711 lookup table"""
    # Build lookup table if not already built
    table = _build_ulaw_table()
    
    # Convert bytes to numpy array of uint8
    ulaw_array = np.frombuffer(ulaw_audio, dtype=np.uint8)
    
    # Use lookup table to convert
    pcm_values = table[ulaw_array]
    
    return pcm_values.tobytes()


def pcm16_to_ulaw(pcm16_audio: bytes) -> bytes:
    """Convert 16-bit PCM audio to μ-law"""
    # Convert bytes to numpy array of int16
    pcm_array = np.frombuffer(pcm16_audio, dtype=np.int16)
    
    # μ-law compression
    # Get sign and magnitude
    sign = np.sign(pcm_array)
    magnitude = np.abs(pcm_array)
    
    # Clip to maximum value
    magnitude = np.clip(magnitude, 0, 32767)
    
    # Add bias
    magnitude = magnitude + 33
    
    # Find exponent (bit position of most significant bit)
    # For simplicity, use log2 approach
    exponent = np.floor(np.log2(magnitude + 1)).astype(np.int16)
    exponent = np.clip(exponent, 0, 7)
    
    # Calculate mantissa
    mantissa = (magnitude >> (exponent + 1)) & 0x0F
    
    # Combine: sign bit (bit 7) | exponent (bits 6-4) | mantissa (bits 3-0)
    ulaw_values = (127 - (exponent << 4) - mantissa)
    ulaw_values[sign < 0] |= 0x80  # Set sign bit
    
    return ulaw_values.astype(np.uint8).tobytes()


def resample_audio(audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample audio using simple linear interpolation"""
    if from_rate == to_rate:
        return audio_data
    
    # Convert bytes to samples (16-bit PCM = 2 bytes per sample)
    num_samples = len(audio_data) // 2
    samples = struct.unpack(f'<{num_samples}h', audio_data)
    
    # Resample
    resampled_samples = []
    ratio = to_rate / from_rate
    
    for i in range(int(num_samples * ratio)):
        source_index = i / ratio
        index1 = int(source_index)
        index2 = min(index1 + 1, num_samples - 1)
        t = source_index - index1
        
        if index1 == index2:
            value = samples[index1]
        else:
            # Linear interpolation
            value = int(samples[index1] * (1 - t) + samples[index2] * t)
        
        resampled_samples.append(max(-32768, min(32767, value)))  # Clamp to int16 range
    
    # Convert back to bytes
    return struct.pack(f'<{len(resampled_samples)}h', *resampled_samples)


class TwilioOpenAIBridge:
    """Bridges Twilio Media Stream to OpenAI Realtime API"""
    
    def __init__(self, twilio_ws, call_sid: str):
        self.twilio_ws = twilio_ws
        self.call_sid = call_sid
        self.openai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.assistant: Optional[RealtimeAssistant] = None
        self.is_running = False
        self.audio_buffer = bytearray()
        self.sequence_number = 0  # Track sequence number for media messages
        self.stream_sid = None  # Store the actual stream SID
        
    async def connect_to_openai(self):
        """Connect to OpenAI Realtime API"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        
        headers = [
            ("Authorization", f"Bearer {OPENAI_API_KEY}"),
            ("OpenAI-Beta", "realtime=v1")
        ]
        
        print(f"[{self.call_sid}] Connecting to OpenAI Realtime API...")
        self.openai_ws = await websockets.connect(
            REALTIME_API_URL,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )
        print(f"[{self.call_sid}] Connected to OpenAI!")
        
        # Initialize assistant
        self.assistant = RealtimeAssistant(OPENAI_API_KEY)
        self.assistant.websocket = self.openai_ws
        self.assistant.is_running = True
        
    async def configure_session(self):
        """Configure OpenAI session - will be done in handle_openai_messages"""
        # Session configuration is handled in handle_openai_messages
        # when we receive the session.created event
        pass
    
    async def send_audio_to_openai(self, ulaw_audio: bytes):
        """Convert Twilio audio and send to OpenAI"""
        if not self.openai_ws or not self.is_running:
            return
        
        try:
            # Convert μ-law to PCM16
            pcm16_audio = ulaw_to_pcm16(ulaw_audio)
            
            # Resample from 8kHz to 24kHz
            resampled_audio = resample_audio(pcm16_audio, TWILIO_SAMPLE_RATE, OPENAI_SAMPLE_RATE)
            
            # Encode to base64
            audio_b64 = base64.b64encode(resampled_audio).decode('utf-8')
            
            # Send to OpenAI
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await self.openai_ws.send(json.dumps(event))
            
        except Exception as e:
            print(f"[{self.call_sid}] Error sending audio to OpenAI: {e}")
    
    async def send_audio_to_twilio(self, pcm16_audio: bytes):
        """Convert OpenAI audio and send to Twilio"""
        if not self.twilio_ws or not self.is_running:
            return
        
        try:
            # Resample from 24kHz to 8kHz
            resampled_audio = resample_audio(pcm16_audio, OPENAI_SAMPLE_RATE, TWILIO_SAMPLE_RATE)
            
            # Convert PCM16 to μ-law
            ulaw_audio = pcm16_to_ulaw(resampled_audio)
            
            # Encode to base64
            audio_b64 = base64.b64encode(ulaw_audio).decode('utf-8')
            
            # Use stream_sid if available, otherwise use call_sid
            stream_id = self.stream_sid or self.call_sid
            
            # Increment sequence number for each media message
            self.sequence_number += 1
            
            # Send to Twilio Media Stream with sequence number
            # Note: "track" field is NOT valid for Media Stream messages to Twilio
            # The track information is only in messages FROM Twilio
            message = {
                "event": "media",
                "streamSid": stream_id,
                "media": {
                    "payload": audio_b64
                },
                "sequenceNumber": str(self.sequence_number)
            }
            # Log the message structure for debugging (first few messages only)
            if self.sequence_number <= 3:
                print(f"[{self.call_sid}] DEBUG: Sending media message to Twilio:", flush=True)
                print(f"  - streamSid: {stream_id}", flush=True)
                print(f"  - sequenceNumber: {self.sequence_number}", flush=True)
                print(f"  - payload length: {len(audio_b64)} chars (base64)", flush=True)
                print(f"  - ulaw audio bytes: {len(ulaw_audio)}", flush=True)
                print(f"  - message JSON (first 200 chars): {json.dumps(message)[:200]}", flush=True)
            
            await self.twilio_ws.send(json.dumps(message))
            print(f"[{self.call_sid}] ✓ Sent {len(ulaw_audio)} bytes of audio to Twilio (seq: {self.sequence_number})", flush=True)
            
        except Exception as e:
            print(f"[{self.call_sid}] Error sending audio to Twilio: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    async def handle_openai_messages(self):
        """Handle messages from OpenAI Realtime API"""
        if not self.openai_ws:
            return
        
        try:
            async for message in self.openai_ws:
                if not self.is_running:
                    break
                
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    
                    if event_type == "session.created":
                        print(f"[{self.call_sid}] Session created!", flush=True)
                        if not self.assistant.session_configured:
                            await self.assistant.configure_session()
                            self.assistant.session_configured = True
                            
                            # Trigger the AI to speak the greeting
                            try:
                                response_event = {
                                    "type": "response.create",
                                    "response": {
                                        "modalities": ["audio", "text"],
                                        "instructions": "Greet the caller in Croatian language ONLY. Say exactly this greeting in Croatian: 'Dobar dan, dobili ste MooveOn asistenta. Kako vam mogu pomoći?' You MUST speak ONLY Croatian, never Spanish, English, or any other language."
                                    }
                                }
                                await self.openai_ws.send(json.dumps(response_event))
                                print(f"[{self.call_sid}] Triggered greeting response", flush=True)
                            except Exception as e:
                                print(f"[{self.call_sid}] Error triggering greeting: {e}", flush=True)
                    
                    elif event_type == "response.audio.delta":
                        # Accumulate audio chunks
                        audio_chunk = data.get("delta", "")
                        if audio_chunk:
                            try:
                                audio_data = base64.b64decode(audio_chunk)
                                self.audio_buffer.extend(audio_data)
                                # Send audio in smaller chunks for better real-time playback
                                # 24kHz = 24000 samples/sec, 16-bit = 2 bytes/sample
                                # 20ms = 480 samples = 960 bytes (smaller chunks = lower latency)
                                if len(self.audio_buffer) >= 960:
                                    await self.send_audio_to_twilio(bytes(self.audio_buffer))
                                    self.audio_buffer.clear()
                                    print(f"[{self.call_sid}] Sent audio chunk to Twilio", flush=True)
                            except Exception as e:
                                print(f"[{self.call_sid}] Error decoding audio: {e}", flush=True)
                    
                    elif event_type == "response.audio.done":
                        # Send any remaining accumulated audio to Twilio
                        if self.audio_buffer:
                            await self.send_audio_to_twilio(bytes(self.audio_buffer))
                            self.audio_buffer.clear()
                            print(f"[{self.call_sid}] Sent final audio chunk to Twilio", flush=True)
                    
                    elif event_type == "response.audio_transcript.done":
                        # Log when audio transcript is done
                        transcript = data.get("transcript", "")
                        print(f"[{self.call_sid}] ASSISTANT SAID: {transcript}", flush=True)
                    
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcription = data.get("transcript", "")
                        if transcription:
                            print(f"[{self.call_sid}] USER SAID: {transcription}")
                    
                    elif event_type == "error":
                        error = data.get("error", {})
                        print(f"[{self.call_sid}] OpenAI Error: {error.get('message', 'Unknown error')}")
                        
                except json.JSONDecodeError:
                    print(f"[{self.call_sid}] Received non-JSON message from OpenAI")
                except Exception as e:
                    print(f"[{self.call_sid}] Error processing OpenAI message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.call_sid}] OpenAI connection closed")
        except Exception as e:
            print(f"[{self.call_sid}] Error handling OpenAI messages: {e}")
    
    async def handle_twilio_messages(self):
        """Handle messages from Twilio Media Stream"""
        try:
            print(f"[{self.call_sid}] Starting Twilio message handler...", flush=True)
            print(f"[{self.call_sid}] WebSocket object: {type(self.twilio_ws)}", flush=True)
            print(f"[{self.call_sid}] WebSocket closed status: {getattr(self.twilio_ws, 'closed', 'unknown')}", flush=True)
            message_count = 0
            last_message_time = None
            async for message in self.twilio_ws:
                if not self.is_running:
                    break
                
                message_count += 1
                # Always log the first 10 messages, then every 20th
                if message_count <= 10 or message_count % 20 == 0:
                    print(f"[{self.call_sid}] Received message #{message_count} from Twilio", flush=True)
                
                # Log the raw message for first few to debug
                if message_count <= 10:
                    print(f"[{self.call_sid}] Raw message #{message_count} (length: {len(message)}): {message[:500]}...", flush=True)
                
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == "media":
                        # Extract audio payload
                        media = data.get("media", {})
                        payload = media.get("payload", "")
                        track = media.get("track", "inbound")  # Usually "inbound" for user's voice
                        
                        print(f"[{self.call_sid}] Received media event - track: {track}, payload length: {len(payload) if payload else 0}", flush=True)
                        
                        if payload:
                            try:
                                # Decode base64 μ-law audio
                                ulaw_audio = base64.b64decode(payload)
                                # Send to OpenAI
                                await self.send_audio_to_openai(ulaw_audio)
                                print(f"[{self.call_sid}] Sent {len(ulaw_audio)} bytes of audio to OpenAI (track: {track})", flush=True)
                            except Exception as e:
                                print(f"[{self.call_sid}] Error processing Twilio audio: {e}", flush=True)
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"[{self.call_sid}] Media event with empty payload", flush=True)
                    
                    elif event_type == "start":
                        # Update call SID and stream SID from start event
                        start_data = data.get("start", {})
                        stream_sid = start_data.get("streamSid") or start_data.get("callSid")
                        call_sid_from_start = start_data.get("callSid")
                        
                        if stream_sid:
                            self.stream_sid = stream_sid
                            print(f"[{self.call_sid}] Stream SID: {stream_sid}", flush=True)
                        
                        if call_sid_from_start and call_sid_from_start != self.call_sid:
                            print(f"[{self.call_sid}] Updating call SID to: {call_sid_from_start}", flush=True)
                            self.call_sid = call_sid_from_start
                        
                        print(f"[{self.call_sid}] Media stream started", flush=True)
                        print(f"[{self.call_sid}] Start event data: {json.dumps(data, indent=2)}", flush=True)
                    
                    elif event_type == "stop":
                        print(f"[{self.call_sid}] Media stream stopped", flush=True)
                        self.is_running = False
                        break
                    
                    elif event_type:
                        print(f"[{self.call_sid}] Received Twilio event: {event_type}", flush=True)
                        # Log full event data for non-media events (first 5)
                        if message_count <= 5:
                            print(f"[{self.call_sid}] Event data: {json.dumps(data, indent=2)}", flush=True)
                        
                except json.JSONDecodeError:
                    print(f"[{self.call_sid}] Received non-JSON message from Twilio: {message[:100]}", flush=True)
                except Exception as e:
                    print(f"[{self.call_sid}] Error processing Twilio message: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
            
            print(f"[{self.call_sid}] Twilio message handler finished. Total messages: {message_count}", flush=True)
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"[{self.call_sid}] Twilio connection closed")
        except Exception as e:
            print(f"[{self.call_sid}] Error handling Twilio messages: {e}")
    
    async def run(self):
        """Run the bridge"""
        try:
            print(f"[{self.call_sid}] Starting bridge...")
            self.is_running = True
            
            # Connect to OpenAI
            await self.connect_to_openai()
            
            # Run both message handlers concurrently
            # Session configuration will happen automatically in handle_openai_messages
            await asyncio.gather(
                self.handle_twilio_messages(),
                self.handle_openai_messages(),
                return_exceptions=True
            )
            
        except Exception as e:
            print(f"[{self.call_sid}] Bridge error: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup connections"""
        print(f"[{self.call_sid}] Cleaning up...")
        self.is_running = False
        
        if self.openai_ws:
            try:
                await self.openai_ws.close()
            except:
                pass
        
        if self.twilio_ws:
            try:
                await self.twilio_ws.close()
            except:
                pass


@app.route('/voice', methods=['POST', 'GET'])
def voice_webhook():
    """Handle incoming Twilio voice call"""
    # Log immediately to confirm webhook received (for debugging timeouts)
    import time
    print(f"\n>>> [WEBHOOK RECEIVED] {request.method} /voice at {time.time()}", flush=True)
    
    # Get call information
    call_sid = request.form.get('CallSid')
    from_number = request.form.get('From')
    to_number = request.form.get('To')
    
    print(f">>> Incoming call:", flush=True)
    print(f"    Call SID: {call_sid}", flush=True)
    print(f"    From: {from_number}", flush=True)
    print(f"    To: {to_number}", flush=True)
    
    # Create TwiML response immediately
    response = VoiceResponse()
    
    # Determine WebSocket URL
    # Use BASE_URL (can be from Render, ngrok, or environment variable)
    base_url = BASE_URL
    
    if not base_url:
        # Try to construct from request (for Render - it provides host in request)
        # For Render, request.host will be something like "your-service.onrender.com"
        if request.host and not request.host.startswith('localhost'):
            scheme = 'https' if request.is_secure or 'onrender.com' in request.host else 'http'
            base_url = f"{scheme}://{request.host}"
            print(f"    Using request host as base URL: {base_url}", flush=True)
        else:
            print("    ERROR: No BASE_URL configured and cannot determine from request!", flush=True)
            response.say("Error: Server configuration missing", language='en-US')
            return str(response), 200, {'Content-Type': 'text/xml'}
    
    # Extract domain and convert to WebSocket URL
    ws_url = base_url.replace('https://', 'wss://').replace('http://', 'ws://')
    if not ws_url.startswith('wss://') and not ws_url.startswith('ws://'):
        ws_url = f"wss://{ws_url}"
    
    media_stream_url = f'{ws_url}/media'
    print(f"    Media Stream URL: {media_stream_url}", flush=True)
    
    # Start Media Stream to get audio from Twilio (bidirectional)
    # Note: Media Streams are bidirectional by default - we receive inbound audio
    # and can send outbound audio via the WebSocket
    # The Stream element doesn't support a 'parameters' attribute
    # Parameters can be passed via URL query string if needed
    start = response.start()
    start.stream(url=media_stream_url)
    
    # Keep the call active - required for Media Streams to work
    # Without this, the call will disconnect after a few seconds
    from twilio.twiml.voice_response import Pause
    response.pause(length=3600)  # Keep call active for up to 1 hour
    
    # AI agent will speak directly, so we don't need a greeting
    # But you can uncomment this if you want a greeting first:
    # response.say("Connecting you to AI assistant...", language='hr-HR')
    
    return str(response), 200, {'Content-Type': 'text/xml'}


@sock.route('/media')
def media_websocket(ws):
    """Handle Twilio Media Stream WebSocket connection via Flask-Sock"""
    call_sid = None
    
    try:
        print(f"New WebSocket connection: {ws}", flush=True)
        
        # Wrap flask-sock's WebSocket to work with our async bridge
        # Convert synchronous ws to async-compatible
        async def handle_async():
            nonlocal call_sid
            call_sid = f"call_{id(ws)}"
            print(f"[{call_sid}] Media stream WebSocket connected", flush=True)
            
            # Create a wrapper to make flask-sock WebSocket work with async bridge
            class FlaskSockWrapper:
                def __init__(self, flask_ws, event_loop):
                    self.flask_ws = flask_ws
                    self.closed = False
                    self._recv_queue = asyncio.Queue()
                    self._loop = event_loop
                    self._recv_lock = asyncio.Lock()
                    
                    # Start thread to continuously read from flask-sock and put in queue
                    import threading
                    def reader_thread():
                        while not self.closed:
                            try:
                                # This blocks until data is available
                                try:
                                    data = self.flask_ws.receive()
                                    if data:
                                        # Put in queue using the event loop
                                        try:
                                            future = asyncio.run_coroutine_threadsafe(
                                                self._recv_queue.put(data),
                                                self._loop
                                            )
                                            future.result(timeout=1.0)  # Wait for it to be queued
                                        except Exception as e:
                                            print(f"[{call_sid}] Error queuing data: {e}", flush=True)
                                except Exception as e:
                                    if not self.closed:
                                        # Check if it's a timeout or connection error
                                        error_str = str(e).lower()
                                        if "timeout" not in error_str and "closed" not in error_str:
                                            print(f"[{call_sid}] Error receiving from flask-sock: {e}", flush=True)
                                    self.closed = True
                                    break
                            except Exception as e:
                                if not self.closed:
                                    print(f"[{call_sid}] Error in reader thread: {e}", flush=True)
                                self.closed = True
                                break
                    
                    self.reader_thread = threading.Thread(target=reader_thread, daemon=True)
                    self.reader_thread.start()
                
                def __aiter__(self):
                    return self
                
                async def __anext__(self):
                    if self.closed:
                        raise StopAsyncIteration
                    try:
                        # Get data from queue - this will wait until data is available
                        while not self.closed:
                            try:
                                # Wait for data (with timeout to check if closed)
                                data = await asyncio.wait_for(self._recv_queue.get(), timeout=1.0)
                                if isinstance(data, str):
                                    return data
                                if data is not None:
                                    return json.dumps(data) if not isinstance(data, str) else data
                            except asyncio.TimeoutError:
                                # Continue waiting if not closed
                                if self.closed:
                                    raise StopAsyncIteration
                                continue
                    except StopAsyncIteration:
                        raise
                    except Exception as e:
                        if self.closed:
                            raise StopAsyncIteration
                        # If there's an error but not closed, try to continue
                        print(f"[{call_sid}] Error in __anext__: {e}", flush=True)
                        raise StopAsyncIteration
                    
                    raise StopAsyncIteration
                
                async def recv(self):
                    try:
                        data = await self._recv_queue.get()
                        if isinstance(data, str):
                            return data
                        return json.dumps(data) if data else None
                    except Exception as e:
                        self.closed = True
                        raise websockets.exceptions.ConnectionClosed(None, None)
                
                async def send(self, data):
                    try:
                        if isinstance(data, str):
                            self.flask_ws.send(data)
                        else:
                            self.flask_ws.send(json.dumps(data))
                    except Exception as e:
                        self.closed = True
                        raise websockets.exceptions.ConnectionClosed(None, None)
                
                async def close(self):
                    self.closed = True
                    try:
                        self.flask_ws.close()
                    except:
                        pass
            
            # Get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            wrapped_ws = FlaskSockWrapper(ws, loop)
            bridge = TwilioOpenAIBridge(wrapped_ws, call_sid)
            await bridge.run()
        
        # Run the async handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(handle_async())
        finally:
            loop.close()
        
    except Exception as e:
        print(f"[{call_sid}] WebSocket error: {e}")
        import traceback
        traceback.print_exc()


async def handle_media_websocket(websocket, path):
    """Handle Twilio Media Stream WebSocket connection"""
    call_sid = None
    
    try:
        print(f"New WebSocket connection: {path}")
        
        # Extract call SID from query parameters if available
        if '?' in path:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(path)
            params = parse_qs(parsed.query)
            if 'name' in params:
                call_sid = params['name'][0]
            elif 'CallSid' in params:
                call_sid = params['CallSid'][0]
        
        if not call_sid:
            call_sid = f"unknown_{id(websocket)}"
            print(f"[{call_sid}] WebSocket connected (will extract call SID from start event)")
        else:
            print(f"[{call_sid}] Media stream WebSocket connected", flush=True)
        
        # Create bridge and run it
        # The bridge will handle all messages including the start event
        bridge = TwilioOpenAIBridge(websocket, call_sid)
        await bridge.run()
        
    except websockets.exceptions.ConnectionClosed:
        print(f"[{call_sid}] WebSocket connection closed")
    except Exception as e:
        print(f"[{call_sid}] WebSocket error: {e}")
        import traceback
        traceback.print_exc()


def start_websocket_server():
    """Start WebSocket server for Twilio Media Streams"""
    port = int(os.getenv("WEBSOCKET_PORT", "8765"))
    print(f"\nStarting WebSocket server on port {port}...")
    print(f"Media Stream WebSocket endpoint: ws://0.0.0.0:{port}/media")
    
    async def run_server():
        async with websockets.serve(handle_media_websocket, "0.0.0.0", port):
            await asyncio.Future()  # run forever
    
    # Run WebSocket server in a separate task
    asyncio.create_task(run_server())


@app.route('/status', methods=['POST'])
def status_webhook():
    """Handle call status updates"""
    call_sid = request.form.get('CallSid')
    call_status = request.form.get('CallStatus')
    
    print(f"\n>>> Call Status Update:")
    print(f"    Call SID: {call_sid}")
    print(f"    Status: {call_status}")
    
    return '', 200


# WebSocket server is now handled by Flask-Sock on the same port as Flask
# No need for separate WebSocket server


def start_background_tasks():
    """Start background async tasks"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_websocket_server())


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Twilio Webhook Server with OpenAI Realtime API")
    print("="*50)
    
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set in environment variables!")
        print("Set it in .env file or as environment variable.")
    
    # Show base URL configuration
    if BASE_URL:
        ws_url = BASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')
        if not ws_url.startswith('wss://') and not ws_url.startswith('ws://'):
            ws_url = f"wss://{ws_url}"
        print(f"Base URL: {BASE_URL}")
        print(f"Media Stream URL: {ws_url}/media")
    else:
        print("WARNING: No BASE_URL configured!")
        print("Set BASE_URL or RENDER_URL in environment variables")
        print("For Render: BASE_URL will be auto-detected from request.host")
        print("For ngrok: Set NGROK_URL in .env file, e.g., NGROK_URL=https://abc123.ngrok.io")
    
    print("\nStarting server...")
    print("  - Flask server on port 5000 (HTTP + WebSocket endpoints)")
    print("  - WebSocket available at: wss://your-ngrok-url/media")
    print("  - Accessible via ngrok: https://your-ngrok-url/voice (HTTP)")
    print("                         wss://your-ngrok-url/media (WebSocket)")
    print("\n" + "="*50 + "\n")
    
    # Run Flask server with eventlet for WebSocket support
    # Flask-Sock works best with eventlet
    try:
        import eventlet
        import eventlet.wsgi
        
        print("Starting server with eventlet for WebSocket support...", flush=True)
        print("Server listening on http://0.0.0.0:5000", flush=True)
        eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5000)), app, log_output=True)
    except ImportError:
        try:
            from gevent import pywsgi
            from geventwebsocket.handler import WebSocketHandler
            
            print("Starting server with gevent for WebSocket support...")
            server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler, log=None)
            server.serve_forever()
        except ImportError:
            print("WARNING: Neither eventlet nor gevent installed, using Flask development server")
            print("WebSocket may not work properly - install eventlet: pip install eventlet")
            # Fallback to Flask development server
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

