# Gunicorn configuration for Render deployment
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker for WebSocket support with eventlet
worker_class = "eventlet"  # Required for Flask-Sock WebSocket support
worker_connections = 1000
timeout = 120  # Increase timeout for WebSocket connections
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'twilio-ai-voice'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (Render handles this automatically)
# keyfile = None
# certfile = None

