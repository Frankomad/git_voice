# Twilio AI Voice Assistant - Render Deployment

AI voice assistant that answers Twilio SIP calls using OpenAI Realtime API.

## Files

- `twilio_webhook_server.py` - Main Flask server that handles Twilio webhooks
- `realtime_assistant.py` - OpenAI Realtime API integration
- `requirements.txt` - Python dependencies
- `render.yaml` - Render.com configuration
- `gunicorn_config.py` - Gunicorn server configuration

## Quick Start for Render

1. **Push this folder to GitHub/GitLab**

2. **Create a Render Web Service:**
   - Go to https://render.com
   - New + → Web Service
   - Connect your repository
   - Render will auto-detect `render.yaml`

3. **Set Environment Variables in Render:**
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. **Deploy** - Render will build and deploy automatically

5. **Get your Render URL** (e.g., `https://your-service.onrender.com`)

6. **Update Twilio Webhook:**
   - Twilio Console → SIP Domains → your domain
   - Set "A CALL COMES IN" to: `https://your-service.onrender.com/voice`
   - Method: POST

## Configuration

- `BASE_URL` - Auto-detected from Render request.host (no need to set)
- `OPENAI_API_KEY` - Required (set in Render environment variables)

## Local Development (with ngrok)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   NGROK_URL=https://your-ngrok-url.ngrok.io
   ```

3. Start server:
   ```bash
   python twilio_webhook_server.py
   ```

4. Start ngrok:
   ```bash
   ngrok http 5000
   ```

5. Update Twilio webhook with ngrok URL

