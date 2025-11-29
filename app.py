#!/usr/bin/env python3
"""
Vehicle Counter Web Application
Flask-based frontend for directional vehicle counting
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import json
import pickle
import threading
import uuid
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Tell Flask it's behind a proxy at /intersection-cam
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Get the script name (proxy prefix) from environment
SCRIPT_NAME = os.getenv('SCRIPT_NAME', '')
if SCRIPT_NAME:
  app.config['APPLICATION_ROOT'] = SCRIPT_NAME

# Configure Flask
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'Abasaleh-12')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000MB max file size

# Create folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

socketio = SocketIO(app,
                    cors_allowed_origins="*",
                    max_http_buffer_size=10 ** 8,
                    async_mode='eventlet',
                    logger=True,
                    engineio_logger=True)

# Store processing sessions
processing_sessions = {}


@app.route('/')
def index():
  """Main upload page"""
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
  """Handle video upload"""
  if 'video' not in request.files:
    return jsonify({'error': 'No video file'}), 400

  file = request.files['video']
  if file.filename == '':
    return jsonify({'error': 'No selected file'}), 400

  # Generate session ID
  session_id = str(uuid.uuid4())

  # Save video
  filename = secure_filename(file.filename)
  video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
  file.save(video_path)

  # Get video info
  cap = cv2.VideoCapture(video_path)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  # Get first frame for zone configuration
  ret, frame = cap.read()
  cap.release()

  if not ret:
    return jsonify({'error': 'Could not read video'}), 400

  # Save first frame
  frame_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_frame.jpg")
  cv2.imwrite(frame_path, frame)

  # Store session info
  processing_sessions[session_id] = {
    'video_path': video_path,
    'frame_path': frame_path,
    'width': width,
    'height': height,
    'fps': fps,
    'total_frames': total_frames,
    'duration': total_frames / fps if fps > 0 else 0,
    'status': 'uploaded'
  }

  return jsonify({
    'session_id': session_id,
    'width': width,
    'height': height,
    'fps': fps,
    'total_frames': total_frames,
    'duration': round(total_frames / fps, 2) if fps > 0 else 0,
    'frame_url': f'/frame/{session_id}'
  })


@app.route('/frame/<session_id>')
def get_frame(session_id):
  """Serve the first frame for zone configuration"""
  if session_id not in processing_sessions:
    return jsonify({'error': 'Session not found'}), 404

  frame_path = processing_sessions[session_id]['frame_path']
  return send_file(frame_path, mimetype='image/jpeg')


@app.route('/configure', methods=['POST'])
def save_configuration():
  """Save zone and line configuration"""
  data = request.json
  session_id = data.get('session_id')

  if session_id not in processing_sessions:
    return jsonify({'error': 'Session not found'}), 404

  # Save configuration
  config_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_config.pkl")

  zone_config = {
    'zones': data.get('zones', []),
    'counting_lines': data.get('counting_lines', [])
  }

  with open(config_path, 'wb') as f:
    pickle.dump(zone_config, f)

  processing_sessions[session_id]['config_path'] = config_path
  processing_sessions[session_id]['zone_config'] = zone_config
  processing_sessions[session_id]['status'] = 'configured'

  return jsonify({'success': True})


@app.route('/process', methods=['POST'])
def start_processing():
  """Start video processing"""
  data = request.json
  session_id = data.get('session_id')

  if session_id not in processing_sessions:
    return jsonify({'error': 'Session not found'}), 404

  if processing_sessions[session_id]['status'] != 'configured':
    return jsonify({'error': 'Session not configured'}), 400

  # Get processing parameters
  params = {
    'model': data.get('model', 'femto_512'),
    'conf_threshold': float(data.get('conf_threshold', 0.25)),
    'min_detections': int(data.get('min_detections', 7)),
    'detection_window': int(data.get('detection_window', 10)),
    'draw_boxes': data.get('draw_boxes', True),
    'device': data.get('device', 'cpu')
  }

  processing_sessions[session_id]['params'] = params
  processing_sessions[session_id]['status'] = 'processing'

  # Start processing in background thread
  thread = threading.Thread(
    target=process_video_worker,
    args=(session_id,)
  )
  thread.daemon = True
  thread.start()

  return jsonify({'success': True})


def process_video_worker(session_id):
  """Background worker for video processing"""
  from counter_worker import VideoProcessor

  session = processing_sessions[session_id]

  # Set model paths
  model_map = {
    'large': ('yolov8_l.py', 'yolov8_l.pth'),
    'medium': ('yolov8_m.py', 'yolov8_m.pth'),
    'small': ('yolov8_s.py', 'yolov8_s.pth'),
    'nano_640': ('yolov8_n_640.py', 'yolov8_n_640.pth'),
    'nano_512': ('yolov8_n_512.py', 'yolov8_n_512.pth'),
    'nano_448': ('yolov8_n_448.py', 'yolov8_n_448.pth'),
    'pico_512': ('yolov8_p_512.py', 'yolov8_p_512.pth'),
    'femto_512': ('yolov8_f_512.py', 'yolov8_f_512.pth'),
    'femto_352': ('yolov8_f_352.py', 'yolov8_f_352.pth'),
  }

  model = session['params']['model']
  config_path, checkpoint_path = model_map.get(model, model_map['femto_512'])

  # Output path
  output_path = os.path.join(
    app.config['OUTPUT_FOLDER'],
    f"{session_id}_output.mp4"
  )

  session['output_path'] = output_path

  # Create processor
  processor = VideoProcessor(
    config_path=f"configs/{config_path}",
    checkpoint_path=f"weights/{checkpoint_path}",
    zone_config=session['zone_config'],
    conf_threshold=session['params']['conf_threshold'],
    device=session['params']['device'],
    min_detections=session['params']['min_detections'],
    detection_window=session['params']['detection_window'],
    draw_boxes=session['params']['draw_boxes'],
    session_id=session_id,
    socketio=socketio
  )

  try:
    # Process video
    stats = processor.process_video(
      video_path=session['video_path'],
      output_path=output_path
    )

    session['stats'] = stats
    session['status'] = 'completed'

    # Send completion event
    socketio.emit('processing_complete', {
      'session_id': session_id,
      'stats': stats
    })

  except Exception as e:
    session['status'] = 'error'
    session['error'] = str(e)
    socketio.emit('processing_error', {
      'session_id': session_id,
      'error': str(e)
    })


@app.route('/download/<session_id>')
def download_video(session_id):
  """Download processed video"""
  if session_id not in processing_sessions:
    return jsonify({'error': 'Session not found'}), 404

  session = processing_sessions[session_id]

  if session['status'] != 'completed':
    return jsonify({'error': 'Processing not complete'}), 400

  return send_file(
    session['output_path'],
    as_attachment=True,
    download_name=f"counted_video_{session_id}.mp4"
  )


@app.route('/stats/<session_id>')
def get_stats(session_id):
  """Get processing statistics"""
  if session_id not in processing_sessions:
    return jsonify({'error': 'Session not found'}), 404

  session = processing_sessions[session_id]

  if 'stats' not in session:
    return jsonify({'error': 'No statistics available'}), 400

  return jsonify(session['stats'])


# For development only
if __name__ == '__main__':
  socketio.run(app, debug=True, host='0.0.0.0', port=8002, allow_unsafe_werkzeug=True)