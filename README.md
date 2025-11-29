# Vehicle Counter Web Application

A professional web-based traffic analysis system with real-time vehicle counting, directional tracking, and interactive zone configuration.

![Vehicle Counter](screenshot.png)

## Features

- 🎥 **Video Upload** - Drag-and-drop or browse to upload traffic videos
- 🎯 **Interactive Zone Configuration** - Draw detection zones and counting lines directly on video frames
- 🚗 **Real-time Processing** - Live preview with WebSocket-powered updates
- 📊 **Analytics Dashboard** - Real-time counts by direction and vehicle class
- 📈 **Visual Reports** - Charts and statistics for traffic analysis
- ⬇️ **Export** - Download processed videos with annotations

## Tech Stack

**Backend:**
- Flask (Web framework)
- Flask-SocketIO (Real-time communication)
- OpenCV (Video processing)
- MMDetection (Object detection)

**Frontend:**
- Vanilla JavaScript (No framework overhead)
- Socket.IO (Real-time updates)
- Chart.js (Data visualization)
- Custom CSS (Industrial/Technical design)

## Installation

### 1. Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- MMDetection and model files (from your existing setup)

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MMDetection (if not already installed)
# Follow: https://mmdetection.readthedocs.io/en/latest/get_started.html
```

### 3. Model Files

Ensure you have the model configuration and checkpoint files in the same directory:
- `yolov8_f_512.py`
- `yolov8_f_512.pth`
- (and other model variants you want to use)

## Running the Application

### Start the Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Usage Workflow

#### Step 1: Upload Video
1. Drag and drop a video file or click to browse
2. Supported formats: MP4, AVI, MOV (max 500MB)
3. Review video information (resolution, duration, FPS)
4. Configure detection parameters:
   - **Model Size**: Choose based on accuracy vs speed needs
   - **Confidence**: Detection confidence threshold (0.1-0.9)
   - **Min Detections**: Minimum detections before counting (3-20)
   - **Detection Window**: Window size for detection tracking (5-30)
   - **Draw Boxes**: Toggle detection box visualization

#### Step 2: Configure Zones
1. **Detection Zones** (Road areas):
   - Click to add points for a polygon
   - Press ENTER to complete the zone
   - Press ESC to cancel current drawing
   - Add multiple zones if needed

2. **Counting Lines** (Directional tracking):
   - Click two points to create a line
   - Enter direction name (e.g., "North→South", "Exit", "Entrance")
   - Add multiple lines for different directions

3. **Controls**:
   - Clear Current: Remove current drawing
   - Reset All: Clear all zones and lines
   - Start Processing: Begin video analysis

#### Step 3: Processing
- Watch real-time preview of video processing
- Monitor live counts by direction
- View vehicle class breakdown
- Track progress (frames processed, FPS)

#### Step 4: Results
- Review final statistics
- Download processed video with annotations
- View charts:
  - Vehicles by direction (bar chart)
  - Vehicles by class (doughnut chart)
- Start new analysis or export data

## Configuration Options

### Model Selection

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| femto_352 | Smallest | Fastest | Good |
| femto_512 | Small | Very Fast | Good |
| pico_512 | Small | Fast | Better |
| nano_448/512/640 | Medium | Balanced | Better |
| small | Large | Slower | High |
| medium | Larger | Slow | Higher |
| large | Largest | Slowest | Best |

### Detection Parameters

**Confidence Threshold (0.1-0.9):**
- Lower = More detections (may include false positives)
- Higher = Fewer detections (more confident)
- Recommended: 0.25-0.35

**Min Detections (3-20):**
- Minimum number of detections before counting a vehicle
- Prevents counting false positives
- Recommended: 7-15 (higher for noisy videos)

**Detection Window (5-30):**
- Number of frames to track detections
- Larger window = more stable tracking
- Recommended: 10-20

## Architecture

### Backend Structure
```
app.py              # Flask application & routes
counter_worker.py   # Video processing with WebSocket updates
directional_counter.py  # Original counter (BoT-SORT tracker)
```

### Frontend Structure
```
templates/
  index.html        # Main application template
static/
  css/
    style.css       # Industrial design theme
  js/
    app.js          # Application logic & WebSocket
```

### Data Flow
1. User uploads video → Flask stores in `uploads/`
2. User configures zones → Saved as pickle file
3. Processing starts → Background thread + WebSocket updates
4. Real-time frames → Encoded as JPEG, sent via WebSocket
5. Completed video → Saved to `outputs/` for download

## API Endpoints

### POST /upload
Upload video file
- **Input**: FormData with video file
- **Output**: Session ID, video info

### GET /frame/<session_id>
Get first frame for zone configuration
- **Output**: JPEG image

### POST /configure
Save zone configuration
- **Input**: JSON with zones and counting lines
- **Output**: Success status

### POST /process
Start video processing
- **Input**: JSON with model and parameters
- **Output**: Success status

### GET /download/<session_id>
Download processed video
- **Output**: MP4 file

### GET /stats/<session_id>
Get processing statistics
- **Output**: JSON with counts and breakdown

## WebSocket Events

### Client → Server
(None - server-driven updates)

### Server → Client

**processing_update**
- Frame image (base64 JPEG)
- Current counts by direction and class
- Frame number

**processing_progress**
- Progress percentage
- Frames processed / total
- Current FPS

**processing_complete**
- Final statistics
- Total counts and breakdowns

**processing_error**
- Error message

## Design Philosophy

The interface uses an **industrial/technical aesthetic** inspired by traffic control systems:

- **Typography**: JetBrains Mono (technical data) + Rajdhani (headings)
- **Colors**: Dark theme with neon accents (traffic light colors)
- **Layout**: Grid-based, information-dense
- **Animations**: Subtle glows and pulses
- **Feedback**: Real-time status indicators

## Troubleshooting

### Video Upload Fails
- Check file size (max 500MB)
- Ensure video format is supported
- Verify disk space

### Processing Errors
- Ensure model files exist
- Check CUDA availability for GPU models
- Verify MMDetection installation

### WebSocket Connection Issues
- Check firewall settings
- Ensure port 5000 is available
- Try different browser

### Slow Processing
- Use smaller model (femto/pico)
- Lower confidence threshold
- Reduce video resolution

## Performance Tips

1. **For Fast Processing**: Use femto_352 or femto_512
2. **For Accuracy**: Use medium or large models
3. **For Real-time**: GPU with CUDA
4. **For Long Videos**: Disable preview or use lower FPS

## License

This project uses MMDetection and follows its license requirements.

## Credits

- Built on MMDetection framework
- Uses BoT-SORT tracking algorithm
- Chart.js for data visualization
- Socket.IO for real-time communication

---

**Developed with attention to both functionality and aesthetics.**
