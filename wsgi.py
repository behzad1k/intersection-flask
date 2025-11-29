#!/usr/bin/env python3
"""
WSGI entry point for Vehicle Counter application
"""

from app import app, socketio

if __name__ == "__main__":
    # For production use with gunicorn
    socketio.run(app)
else:
    # This is the application object that gunicorn will use
    application = app
