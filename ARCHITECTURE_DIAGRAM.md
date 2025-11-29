# Application Flow Diagram

## User Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                      VEHICLE COUNTER WEB APP                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  1. UPLOAD      │
│  ─────────      │
│  • Drop video   │──┐
│  • Configure    │  │
│  • Parameters   │  │
└─────────────────┘  │
                     │
                     ▼
                ┌─────────────────┐
                │  Flask Server   │
                │  (app.py)       │
                │                 │
                │  • Save video   │
                │  • Extract      │
                │    first frame  │
                │  • Create       │
                │    session ID   │
                └─────────────────┘
                     │
                     ▼
┌─────────────────┐
│  2. CONFIGURE   │
│  ─────────      │
│  • Draw zones   │◄─┐
│  • Draw lines   │  │
│  • Name dirs    │  │
└─────────────────┘  │
                     │
                     ▼
                ┌─────────────────┐
                │  Save Config    │
                │  (pickle)       │
                └─────────────────┘
                     │
                     ▼
┌─────────────────┐
│  3. PROCESS     │
│  ─────────      │◄─────────────────┐
│  • Live view    │                  │
│  • Real-time    │                  │
│    counts       │                  │
└─────────────────┘                  │
        ▲                            │
        │                            │
        │  WebSocket Updates         │
        │  ─────────────────         │
        │  • Frame images            │
        │  • Progress %              │
        │  • Count data              │
        │                            │
        │                            │
   ┌────┴──────────────┐             │
   │  Background       │             │
   │  Processing       │             │
   │  (counter_worker) │─────────────┘
   │                   │
   │  • Load model     │
   │  • Process frames │
   │  • Track vehicles │
   │  • Count crossing │
   │  • Save output    │
   └───────────────────┘
        │
        ▼
┌─────────────────┐
│  4. RESULTS     │
│  ─────────      │
│  • Statistics   │
│  • Charts       │
│  • Download     │
└─────────────────┘
```

## Data Flow

```
┌──────────┐
│  Video   │
│  Upload  │
└────┬─────┘
     │
     ▼
┌──────────────────┐
│  uploads/        │
│  session.mp4     │
│  session.jpg     │
│  session.pkl     │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Processing      │
│  ─────────       │
│  Frame-by-frame  │
│  detection &     │
│  tracking        │
└────┬─────────────┘
     │
     ├──► WebSocket ──► Browser ──► Live Update
     │
     ▼
┌──────────────────┐
│  outputs/        │
│  session_out.mp4 │
└──────────────────┘
```

## Component Interaction

```
┌─────────────────────────────────────────────────────┐
│                    BROWSER                          │
│  ┌───────────────────────────────────────────────┐  │
│  │  index.html (Single Page App)                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │  Upload  │  │  Canvas  │  │  Charts  │   │  │
│  │  │  Section │  │  Drawing │  │  Display │   │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘   │  │
│  └───────┼─────────────┼─────────────┼─────────┘  │
│          │             │             │            │
│  ┌───────┴─────────────┴─────────────┴─────────┐  │
│  │  app.js (Application Logic)                 │  │
│  │  • File handling                             │  │
│  │  • Zone drawing                              │  │
│  │  • WebSocket client                          │  │
│  │  • Chart generation                          │  │
│  └──────────────────┬───────────────────────────┘  │
└─────────────────────┼───────────────────────────────┘
                      │
                      │ HTTP / WebSocket
                      │
┌─────────────────────┼───────────────────────────────┐
│                     ▼                                │
│  ┌────────────────────────────────────────────┐     │
│  │  Flask Server (app.py)                    │     │
│  │  ┌──────────────────────────────────────┐ │     │
│  │  │  Routes                              │ │     │
│  │  │  • /upload   POST                    │ │     │
│  │  │  • /configure POST                   │ │     │
│  │  │  • /process  POST                    │ │     │
│  │  │  • /download GET                     │ │     │
│  │  └──────────────────────────────────────┘ │     │
│  │  ┌──────────────────────────────────────┐ │     │
│  │  │  SocketIO                            │ │     │
│  │  │  • processing_update                 │ │     │
│  │  │  • processing_progress               │ │     │
│  │  │  • processing_complete               │ │     │
│  │  └──────────────────────────────────────┘ │     │
│  └────────────────────┬───────────────────────┘     │
│                       │                             │
│                       ▼                             │
│  ┌────────────────────────────────────────────┐     │
│  │  Background Thread                        │     │
│  │  ┌──────────────────────────────────────┐ │     │
│  │  │  counter_worker.py                   │ │     │
│  │  │  • Load model                        │ │     │
│  │  │  • Process frames                    │ │     │
│  │  │  • Send updates via SocketIO         │ │     │
│  │  └─────────────┬────────────────────────┘ │     │
│  └────────────────┼───────────────────────────┘     │
│                   │                                 │
│                   ▼                                 │
│  ┌────────────────────────────────────────────┐     │
│  │  directional_counter.py                   │     │
│  │  ┌──────────────────────────────────────┐ │     │
│  │  │  BoTSORTTracker                      │ │     │
│  │  │  • Object tracking                   │ │     │
│  │  │  • Zone detection                    │ │     │
│  │  │  • Line crossing                     │ │     │
│  │  │  • Vehicle counting                  │ │     │
│  │  └──────────────────────────────────────┘ │     │
│  └────────────────────────────────────────────┘     │
│                                                     │
│                   SERVER                            │
└─────────────────────────────────────────────────────┘
```

## WebSocket Communication

```
Browser                          Server
  │                                │
  │  Connect                       │
  ├────────────────────────────────>
  │                                │
  │  Start Processing (HTTP POST)  │
  ├────────────────────────────────>
  │                                │
  │                                │ Start Background
  │                                │ Thread
  │                                │
  │  ◄─ processing_update          │
  │     (frame + counts)           │
  │  ◄─────────────────────────────┤
  │                                │
  │  ◄─ processing_progress        │
  │     (% complete)               │
  │  ◄─────────────────────────────┤
  │                                │
  │  ◄─ processing_update          │
  │     (frame + counts)           │
  │  ◄─────────────────────────────┤
  │                                │
  │  ...every 0.5 seconds...       │
  │                                │
  │  ◄─ processing_complete        │
  │     (final stats)              │
  │  ◄─────────────────────────────┤
  │                                │
  │  Download Video (HTTP GET)     │
  ├────────────────────────────────>
  │                                │
  │  ◄─ video file                 │
  │  ◄─────────────────────────────┤
```

## Session Lifecycle

```
┌──────────────┐
│ Video Upload │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Generate Session ID  │
│ (UUID)              │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Store Session Data   │
│ • video_path        │
│ • frame_path        │
│ • video_info        │
│ • status: uploaded  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ User Configures      │
│ • zones             │
│ • counting_lines    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Update Session       │
│ • config_path       │
│ • zone_config       │
│ • status: configured│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Start Processing     │
│ • params            │
│ • status: processing│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Process Complete     │
│ • output_path       │
│ • stats             │
│ • status: completed │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ User Downloads       │
│ • Processed video   │
│ • Statistics        │
└──────────────────────┘
```

## File Organization

```
Server Filesystem
│
├── uploads/                    (Temporary storage)
│   ├── {session_id}_video.mp4     ← Original video
│   ├── {session_id}_frame.jpg     ← First frame
│   └── {session_id}_config.pkl    ← Zone config
│
├── outputs/                    (Final results)
│   └── {session_id}_output.mp4    ← Processed video
│
└── Session Memory              (Runtime only)
    └── processing_sessions[session_id]
        ├── video_path
        ├── frame_path
        ├── config_path
        ├── output_path
        ├── zone_config
        ├── params
        ├── stats
        └── status
```

---

**Legend:**
- ┌─┐ │ └─┘  = Boxes and connections
- ──► = Data flow direction
- ◄── = Response/callback
- • = List item or bullet point
