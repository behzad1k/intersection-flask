#!/usr/bin/env python3
"""
Video Processor with Real-time WebSocket Updates
Modified version that sends progress updates to frontend
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import base64
import time


class VideoProcessor:
    """Video processor with WebSocket progress updates"""
    
    VEHICLE_CLASSES = {
        0: 'bicycle',
        1: 'motorcycle',
        2: 'car',
        3: 'transporter',
        4: 'bus',
        5: 'truck',
        6: 'trailer',
        7: 'unknown',
        8: 'mask'
    }
    
    def __init__(self, config_path, checkpoint_path, zone_config,
                 conf_threshold=0.25, device='cpu', min_detections=7,
                 detection_window=10, draw_boxes=True, session_id=None,
                 socketio=None):
        """Initialize processor with WebSocket support"""
        
        self.conf_threshold = conf_threshold
        self.device = device
        self.min_detections = min_detections
        self.detection_window = detection_window
        self.draw_boxes = draw_boxes
        self.session_id = session_id
        self.socketio = socketio
        
        # Zone configuration
        self.detection_zones = [np.array(zone) for zone in zone_config['zones']]
        self.counting_lines = zone_config['counting_lines']
        
        # Load model
        from directional_counter import BoTSORTTracker
        from mmdet.apis import init_detector, inference_detector

        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.inference_detector = inference_detector
        self.tracker = BoTSORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Tracking state
        self.tracked_objects = {}
        self.directional_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.frame_count = 0
        
        # For progress updates
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Send updates every 0.5 seconds
        
        # Colors
        self.colors = {
            'bicycle': (0, 255, 255),
            'motorcycle': (255, 0, 255),
            'car': (0, 255, 0),
            'transporter': (255, 165, 0),
            'bus': (255, 0, 0),
            'truck': (0, 165, 255),
            'trailer': (128, 0, 128),
            'unknown': (128, 128, 128)
        }
    
    def point_in_zone(self, point, zone):
        """Check if point is inside polygon zone"""
        return cv2.pointPolygonTest(zone, point, False) >= 0
    
    def line_intersection(self, p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects line segment p3-p4"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def send_update(self, frame, force=False):
        """Send progress update via WebSocket"""
        if not self.socketio or not self.session_id:
            return
        
        current_time = time.time()
        
        # Only send updates at specified interval unless forced
        if not force and current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare counts data
        counts_data = {
            'total': sum(self.total_counts.values()),
            'by_direction': dict(self.total_counts),
            'by_class': {}
        }
        
        # Add class breakdown per direction
        for direction in self.counting_lines:
            dir_name = direction['direction']
            counts_data['by_class'][dir_name] = dict(
                self.directional_counts[dir_name]
            )
        
        # Send update
        self.socketio.emit('processing_update', {
            'session_id': self.session_id,
            'frame': frame_base64,
            'frame_number': self.frame_count,
            'counts': counts_data
        })
    
    def process_frame(self, frame):
        """Process single frame with real-time updates"""
        
        # Run inference
        result = self.inference_detector(self.model, frame)
        
        # Extract detections
        detections = []
        if hasattr(result, 'pred_instances'):
            pred_instances = result.pred_instances
            
            if hasattr(pred_instances, 'bboxes'):
                boxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score >= self.conf_threshold and label != 8:
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2
                        
                        in_zone = False
                        for zone in self.detection_zones:
                            if self.point_in_zone((cx, cy), zone):
                                in_zone = True
                                break
                        
                        if in_zone:
                            detection = np.concatenate([box, [score, label]])
                            detections.append(detection)
        
        # Update tracker
        tracked_detections = self.tracker.update(detections)
        
        # Process tracked detections
        for track_id, detection in tracked_detections:
            x1, y1, x2, y2, conf, cls_id = detection
            
            # Get most common class
            most_common_cls_id = self.tracker.get_most_common_class(track_id)
            if most_common_cls_id is not None:
                cls_id = most_common_cls_id
            else:
                cls_id = int(cls_id)
            
            class_name = self.VEHICLE_CLASSES.get(cls_id, f'class_{cls_id}')
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Initialize tracking state
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = {
                    'class': class_name,
                    'detection_history': deque(maxlen=self.detection_window),
                    'counted_directions': set(),
                    'first_seen': self.frame_count,
                    'prev_position': (cx, cy)
                }
            
            # Update class
            self.tracked_objects[track_id]['class'] = class_name
            self.tracked_objects[track_id]['detection_history'].append(True)
            
            # Check for line crossings
            prev_pos = self.tracked_objects[track_id]['prev_position']
            curr_pos = (cx, cy)
            
            detection_count = sum(self.tracked_objects[track_id]['detection_history'])
            
            if detection_count >= self.min_detections:
                for line_data in self.counting_lines:
                    direction = line_data['direction']
                    
                    if direction in self.tracked_objects[track_id]['counted_directions']:
                        continue
                    
                    line_p1 = tuple(line_data['points'][0])
                    line_p2 = tuple(line_data['points'][1])
                    
                    if self.line_intersection(prev_pos, curr_pos, line_p1, line_p2):
                        self.tracked_objects[track_id]['counted_directions'].add(direction)
                        self.directional_counts[direction][class_name] += 1
                        self.total_counts[direction] += 1
            
            # Update position
            self.tracked_objects[track_id]['prev_position'] = curr_pos
            
            # Draw if enabled
            if self.draw_boxes:
                color = self.colors.get(class_name, (255, 255, 255))
                counted_any = len(self.tracked_objects[track_id]['counted_directions']) > 0
                
                thickness = 3 if counted_any else 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            color, thickness)
                
                # Draw label
                label = f"{class_name} #{track_id}"
                if counted_any:
                    label += " ✓"
                else:
                    label += f" ({detection_count}/{self.min_detections})"
                
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - lh - 10),
                            (int(x1) + lw, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
        
        # Update detection history for non-detected objects
        current_track_ids = set(tid for tid, _ in tracked_detections)
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in current_track_ids:
                self.tracked_objects[track_id]['detection_history'].append(False)
        
        # Draw zones and lines
        self._draw_zones_and_lines(frame)
        
        self.frame_count += 1
        return frame
    
    def _draw_zones_and_lines(self, frame):
        """Draw detection zones and counting lines"""
        # Draw zones
        overlay = frame.copy()
        for zone in self.detection_zones:
            cv2.polylines(frame, [zone], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [zone], (0, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        # Draw counting lines
        for line_data in self.counting_lines:
            pts = line_data['points']
            direction = line_data['direction']
            
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 3)
            cv2.circle(frame, tuple(pts[0]), 6, (255, 0, 0), -1)
            cv2.circle(frame, tuple(pts[1]), 6, (255, 0, 0), -1)
            
            # Label
            mid_x = (pts[0][0] + pts[1][0]) // 2
            mid_y = (pts[0][1] + pts[1][1]) // 2
            
            label = f"{direction}: {self.total_counts[direction]}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (mid_x - 5, mid_y - lh - 5),
                        (mid_x + lw + 5, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, label, (mid_x, mid_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_video(self, video_path, output_path):
        """Process video with real-time updates"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated = self.process_frame(frame)
            
            # Write to output
            writer.write(annotated)
            
            # Send update
            self.send_update(annotated)
            
            # Send progress
            if self.socketio and self.session_id:
                progress = (self.frame_count / total_frames) * 100
                self.socketio.emit('processing_progress', {
                    'session_id': self.session_id,
                    'frame_number': self.frame_count,
                    'total_frames': total_frames,
                    'progress': round(progress, 2),
                    'fps': self.frame_count / (time.time() - start_time) if time.time() > start_time else 0
                })
        
        cap.release()
        writer.release()
        
        # Send final update
        if self.socketio and self.session_id:
            self.send_update(annotated, force=True)
        
        return {
            'total_all': sum(self.total_counts.values()),
            'directional_totals': dict(self.total_counts),
            'directional_breakdown': {
                direction: dict(counts)
                for direction, counts in self.directional_counts.items()
            },
            'timestamp': datetime.now().isoformat(),
            'frames_processed': self.frame_count
        }
