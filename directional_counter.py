#!/usr/bin/env python3
"""
Directional Vehicle Counter with Zone Support
- Counts vehicles from each direction to each direction
- Zone-based detection to avoid sidewalk false positives
- Line crossing detection for directional counting
- Interactive zone configuration
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import argparse
import json
from pathlib import Path
import pickle


class BoTSORTTracker:
  """Simplified BoT-SORT tracker implementation"""

  def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.tracks = {}
    self.next_id = 0
    self.frame_count = 0

    # Enhanced tracking parameters
    self.iou_threshold_low = iou_threshold * 0.5  # Lower threshold for re-identification
    self.max_distance_threshold = 400  # Base max pixels between frames (increased from 150)
    self.velocity_multiplier = 2.5  # Multiply by velocity for fast vehicles

  def update(self, detections):
    """Update tracks with new detections"""
    self.frame_count += 1

    # Age existing tracks
    for track_id in list(self.tracks.keys()):
      self.tracks[track_id]['age'] += 1
      if self.tracks[track_id]['age'] > self.max_age:
        del self.tracks[track_id]

    if len(detections) == 0:
      return []

    # Match detections to existing tracks
    matched_tracks = []
    unmatched_detections = []

    if len(self.tracks) == 0:
      unmatched_detections = list(range(len(detections)))
    else:
      # Enhanced matching with motion prediction
      iou_matrix = np.zeros((len(detections), len(self.tracks)))
      distance_matrix = np.zeros((len(detections), len(self.tracks)))
      track_ids = list(self.tracks.keys())

      for d_idx, det in enumerate(detections):
        det_box = det[:4]
        for t_idx, track_id in enumerate(track_ids):
          # Use predicted position if available
          predicted_box = self._predict_next_position(track_id)
          if predicted_box is None:
            predicted_box = self.tracks[track_id]['box']

          # Calculate IoU with predicted position
          iou_matrix[d_idx, t_idx] = self._calculate_iou(det_box, predicted_box)

          # Also calculate center distance for backup matching
          distance_matrix[d_idx, t_idx] = self._calculate_center_distance(
            det_box, predicted_box)

      # Stage 1: High confidence matching (normal IoU threshold)
      matched_det_indices = set()
      matched_track_indices = set()
      matches = []

      for d_idx in range(len(detections)):
        for t_idx in range(len(track_ids)):
          if iou_matrix[d_idx, t_idx] > self.iou_threshold:
            matches.append((iou_matrix[d_idx, t_idx], d_idx, t_idx))

      matches.sort(reverse=True)

      for _, d_idx, t_idx in matches:
        if d_idx not in matched_det_indices and t_idx not in matched_track_indices:
          track_id = track_ids[t_idx]
          matched_tracks.append((track_id, d_idx))
          matched_det_indices.add(d_idx)
          matched_track_indices.add(t_idx)

      # Stage 2: Re-identification for lost tracks
      # Use lower IoU threshold AND distance for unmatched detections
      remaining_detections = [i for i in range(len(detections))
                              if i not in matched_det_indices]
      remaining_tracks = [i for i in range(len(track_ids))
                          if i not in matched_track_indices]

      recovery_matches = []
      for d_idx in remaining_detections:
        for t_idx in remaining_tracks:
          track_id = track_ids[t_idx]

          # Check if this track was recently active (not aged too much)
          if self.tracks[track_id]['age'] < 5:  # Recently lost
            iou = iou_matrix[d_idx, t_idx]
            dist = distance_matrix[d_idx, t_idx]

            # Calculate velocity-adaptive distance threshold
            velocity = self._get_velocity_magnitude(track_id)
            adaptive_threshold = self.max_distance_threshold + (velocity * self.velocity_multiplier)

            # Accept if either:
            # 1. Low IoU but close distance (vehicle moved/turned)
            #    - Use ADAPTIVE threshold for fast vehicles
            # 2. Moderate IoU (slightly below normal threshold)
            if (iou > self.iou_threshold_low and dist < adaptive_threshold) or \
              (iou > self.iou_threshold * 0.7):
              # Score based on combination of IoU and proximity
              # For fast vehicles, distance is less penalized
              distance_score = 1.0 - min(dist / adaptive_threshold, 1.0)
              score = iou + distance_score * 0.3
              recovery_matches.append((score, d_idx, t_idx, velocity))

      recovery_matches.sort(reverse=True)

      for score, d_idx, t_idx, velocity in recovery_matches:
        if d_idx not in matched_det_indices and t_idx not in matched_track_indices:
          track_id = track_ids[t_idx]
          matched_tracks.append((track_id, d_idx))
          matched_det_indices.add(d_idx)
          matched_track_indices.add(t_idx)

      unmatched_detections = [i for i in range(len(detections))
                              if i not in matched_det_indices]

    # Update matched tracks
    results = []
    for track_id, det_idx in matched_tracks:
      detection = detections[det_idx]
      self.tracks[track_id]['box'] = detection[:4]

      # Track class history instead of just current class
      current_class = int(detection[5])
      if 'class_history' not in self.tracks[track_id]:
        self.tracks[track_id]['class_history'] = []
      self.tracks[track_id]['class_history'].append(current_class)

      # Keep class history manageable (last 50 detections)
      if len(self.tracks[track_id]['class_history']) > 50:
        self.tracks[track_id]['class_history'].pop(0)

      # Store current class (will be overridden by most common later)
      self.tracks[track_id]['class'] = current_class
      self.tracks[track_id]['conf'] = detection[4]
      self.tracks[track_id]['age'] = 0
      self.tracks[track_id]['hits'] += 1

      # Store center point for trajectory
      cx = (detection[0] + detection[2]) / 2
      cy = (detection[1] + detection[3]) / 2
      self.tracks[track_id]['trajectory'].append((cx, cy))

      results.append((track_id, detection))

    # Create new tracks
    for det_idx in unmatched_detections:
      detection = detections[det_idx]
      track_id = self.next_id
      self.next_id += 1

      cx = (detection[0] + detection[2]) / 2
      cy = (detection[1] + detection[3]) / 2

      self.tracks[track_id] = {
        'box': detection[:4],
        'class': int(detection[5]),
        'class_history': [int(detection[5])],  # Initialize class history
        'conf': detection[4],
        'age': 0,
        'hits': 1,
        'trajectory': deque([(cx, cy)], maxlen=50)
      }
      results.append((track_id, detection))

    return results

  def _calculate_iou(self, box1, box2):
    """Calculate Intersection over Union"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max < xi_min or yi_max < yi_min:
      return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0

  def _calculate_center_distance(self, box1, box2):
    """Calculate distance between box centers"""
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

  def _predict_next_position(self, track_id):
    """Predict next position based on trajectory"""
    if track_id not in self.tracks:
      return None

    trajectory = list(self.tracks[track_id]['trajectory'])
    if len(trajectory) < 2:
      return self.tracks[track_id]['box']

    # Use last 2 positions to predict next
    last_pos = trajectory[-1]
    prev_pos = trajectory[-2]

    # Calculate velocity
    vx = last_pos[0] - prev_pos[0]
    vy = last_pos[1] - prev_pos[1]

    # Predict center
    pred_cx = last_pos[0] + vx
    pred_cy = last_pos[1] + vy

    # Get box size from last detection
    box = self.tracks[track_id]['box']
    w = box[2] - box[0]
    h = box[3] - box[1]

    # Return predicted box
    return [pred_cx - w / 2, pred_cy - h / 2, pred_cx + w / 2, pred_cy + h / 2]

  def _get_velocity_magnitude(self, track_id):
    """Calculate velocity magnitude for adaptive distance threshold"""
    if track_id not in self.tracks:
      return 0.0

    trajectory = list(self.tracks[track_id]['trajectory'])
    if len(trajectory) < 2:
      return 0.0

    # Calculate velocity from last few positions
    velocities = []
    for i in range(len(trajectory) - 1, max(0, len(trajectory) - 5), -1):
      if i > 0:
        vx = trajectory[i][0] - trajectory[i - 1][0]
        vy = trajectory[i][1] - trajectory[i - 1][1]
        velocities.append(np.sqrt(vx ** 2 + vy ** 2))

    # Return average velocity
    return np.mean(velocities) if velocities else 0.0

  def get_most_common_class(self, track_id):
    """Get the most frequently detected class for a track"""
    if track_id not in self.tracks:
      return None

    if 'class_history' not in self.tracks[track_id] or not self.tracks[track_id]['class_history']:
      return self.tracks[track_id].get('class', None)

    # Count occurrences of each class
    class_counts = {}
    for cls in self.tracks[track_id]['class_history']:
      class_counts[cls] = class_counts.get(cls, 0) + 1

    # Return the most common class
    most_common_class = max(class_counts, key=class_counts.get)
    return most_common_class


class ZoneConfigTool:
  """Interactive tool for configuring detection zones and counting lines"""

  def __init__(self, frame):
    self.frame = frame.copy()
    self.original = frame.copy()
    self.zones = []
    self.counting_lines = []
    self.current_points = []
    self.mode = 'zone'  # 'zone' or 'line'

  def mouse_callback(self, event, x, y, flags, param):
    """Handle mouse events for drawing zones and lines"""
    if event == cv2.EVENT_LBUTTONDOWN:
      self.current_points.append((x, y))

      if self.mode == 'zone':
        # Draw points
        cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)

        # Draw lines between points
        if len(self.current_points) > 1:
          cv2.line(self.frame, self.current_points[-2],
                   self.current_points[-1], (0, 255, 0), 2)

      elif self.mode == 'line':
        # For lines, we only need 2 points
        cv2.circle(self.frame, (x, y), 5, (255, 0, 0), -1)

        if len(self.current_points) == 2:
          cv2.line(self.frame, self.current_points[0],
                   self.current_points[1], (255, 0, 0), 2)

      cv2.imshow('Zone Configuration', self.frame)

  def configure(self):
    """Interactive configuration"""
    cv2.namedWindow('Zone Configuration')
    cv2.setMouseCallback('Zone Configuration', self.mouse_callback)

    print("\n" + "=" * 70)
    print("ZONE CONFIGURATION")
    print("=" * 70)
    print("\nMODE 1: Define Detection Zones (Road Areas)")
    print("  - Click to add points for polygon zone")
    print("  - Press 'z' to finish current zone")
    print("  - Press 'n' when done with zones")
    print("\nMODE 2: Define Counting Lines")
    print("  - Click two points to create a counting line")
    print("  - Enter direction name when prompted")
    print("  - Press 'n' when done with lines")
    print("\nOther commands:")
    print("  - Press 'c' to clear current drawing")
    print("  - Press 'r' to reset everything")
    print("  - Press 'q' to quit and save")
    print("=" * 70)

    cv2.imshow('Zone Configuration', self.frame)

    while True:
      key = cv2.waitKey(1) & 0xFF

      if key == ord('z') and self.mode == 'zone':
        # Finish current zone
        if len(self.current_points) >= 3:
          self.zones.append(np.array(self.current_points))
          # Draw filled polygon
          overlay = self.frame.copy()
          cv2.fillPoly(overlay, [np.array(self.current_points)],
                       (0, 255, 0))
          cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
          cv2.polylines(self.frame, [np.array(self.current_points)],
                        True, (0, 255, 0), 2)

          print(f"✓ Zone {len(self.zones)} saved with {len(self.current_points)} points")
          self.current_points = []
          cv2.imshow('Zone Configuration', self.frame)
        else:
          print("⚠ Need at least 3 points for a zone")

      elif key == ord('n'):
        if self.mode == 'zone':
          # Switch to line mode
          self.mode = 'line'
          self.current_points = []
          print("\n→ Switched to COUNTING LINE mode")
          print("  Click two points to create a counting line")
          print("  You'll be prompted to enter direction name")
        elif self.mode == 'line':
          # Finish current line if any
          if len(self.current_points) == 2:
            direction = input("\nEnter direction name for this line (e.g., 'North->South'): ")
            self.counting_lines.append({
              'points': self.current_points.copy(),
              'direction': direction
            })
            print(f"✓ Counting line '{direction}' saved")
            self.current_points = []

          # Ask if done
          response = input("\nAdd another counting line? (y/n): ")
          if response.lower() != 'y':
            break

      elif key == ord('c'):
        # Clear current drawing
        self.current_points = []
        self.frame = self.original.copy()

        # Redraw saved zones
        for zone in self.zones:
          overlay = self.frame.copy()
          cv2.fillPoly(overlay, [zone], (0, 255, 0))
          cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
          cv2.polylines(self.frame, [zone], True, (0, 255, 0), 2)

        # Redraw saved lines
        for line_data in self.counting_lines:
          pts = line_data['points']
          cv2.line(self.frame, pts[0], pts[1], (255, 0, 0), 2)
          cv2.circle(self.frame, pts[0], 5, (255, 0, 0), -1)
          cv2.circle(self.frame, pts[1], 5, (255, 0, 0), -1)

          # Draw label
          mid_x = (pts[0][0] + pts[1][0]) // 2
          mid_y = (pts[0][1] + pts[1][1]) // 2
          cv2.putText(self.frame, line_data['direction'], (mid_x, mid_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Zone Configuration', self.frame)
        print("✓ Cleared current drawing")

      elif key == ord('r'):
        # Reset everything
        self.zones = []
        self.counting_lines = []
        self.current_points = []
        self.frame = self.original.copy()
        self.mode = 'zone'
        cv2.imshow('Zone Configuration', self.frame)
        print("✓ Reset all zones and lines")

      elif key == ord('q'):
        break

      # Auto-save line when 2 points selected
      if self.mode == 'line' and len(self.current_points) == 2:
        direction = input("\nEnter direction name for this line (e.g., 'North->South'): ")
        self.counting_lines.append({
          'points': self.current_points.copy(),
          'direction': direction
        })

        # Draw label
        mid_x = (self.current_points[0][0] + self.current_points[1][0]) // 2
        mid_y = (self.current_points[0][1] + self.current_points[1][1]) // 2
        cv2.putText(self.frame, direction, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Zone Configuration', self.frame)

        print(f"✓ Counting line '{direction}' saved")
        self.current_points = []

    cv2.destroyAllWindows()

    config = {
      'zones': [zone.tolist() for zone in self.zones],
      'counting_lines': self.counting_lines
    }

    print(f"\n✓ Configuration complete:")
    print(f"  - {len(self.zones)} detection zones")
    print(f"  - {len(self.counting_lines)} counting lines")

    return config


class DirectionalCounter:
  """Enhanced vehicle counter with zone support and directional counting"""

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
               detection_window=10):
    """
    Args:
        config_path: Path to model config
        checkpoint_path: Path to checkpoint
        zone_config: Zone configuration dict with 'zones' and 'counting_lines'
        conf_threshold: Confidence threshold
        device: 'cuda:0' or 'cpu'
        min_detections: Minimum detections to count
        detection_window: Detection window size
    """
    self.conf_threshold = conf_threshold
    self.device = device
    self.min_detections = min_detections
    self.detection_window = detection_window

    # Zone configuration
    self.detection_zones = [np.array(zone) for zone in zone_config['zones']]
    self.counting_lines = zone_config['counting_lines']

    print(f"Loading model...")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Detection zones: {len(self.detection_zones)}")
    print(f"  Counting lines: {len(self.counting_lines)}")

    try:
      from mmdet.apis import init_detector, inference_detector
      self.model = init_detector(config_path, checkpoint_path, device=device)
      self.inference_detector = inference_detector
      print(f"✓ Model loaded successfully!")
    except Exception as e:
      print(f"\nERROR: Failed to load model!")
      print(f"Details: {e}")
      raise

    # Initialize tracker
    self.tracker = BoTSORTTracker(max_age=30, min_hits=3, iou_threshold=0.3)

    # Tracking state
    self.tracked_objects = {}
    self.directional_counts = defaultdict(lambda: defaultdict(int))  # direction -> class -> count
    self.total_counts = defaultdict(int)  # direction -> total
    self.frame_count = 0

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

  def process_frame(self, frame):
    """Process single frame"""

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
          if score >= self.conf_threshold and label != 8:  # Skip mask class
            # Check if detection is in any zone
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

      # Get the most common class from tracking history
      # This prevents single-frame misclassifications from affecting the count
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

      # Update class to most common (handles class changes over time)
      self.tracked_objects[track_id]['class'] = class_name

      # Add to detection history
      self.tracked_objects[track_id]['detection_history'].append(True)

      # Check for line crossings
      prev_pos = self.tracked_objects[track_id]['prev_position']
      curr_pos = (cx, cy)

      detection_count = sum(self.tracked_objects[track_id]['detection_history'])

      # Only check crossings if minimum detections met
      if detection_count >= self.min_detections:
        for line_data in self.counting_lines:
          direction = line_data['direction']

          # Skip if already counted for this direction
          if direction in self.tracked_objects[track_id]['counted_directions']:
            continue

          line_p1 = tuple(line_data['points'][0])
          line_p2 = tuple(line_data['points'][1])

          # Check if trajectory crossed the line
          if self.line_intersection(prev_pos, curr_pos, line_p1, line_p2):
            # Count it!
            self.tracked_objects[track_id]['counted_directions'].add(direction)
            self.directional_counts[direction][class_name] += 1
            self.total_counts[direction] += 1

            print(f"✓ Counted: {class_name} (ID: {track_id}) - "
                  f"{direction} - Total: {self.total_counts[direction]}")

      # Update position
      self.tracked_objects[track_id]['prev_position'] = curr_pos

      # Draw bounding box
      color = self.colors.get(class_name, (255, 255, 255))
      counted_any = len(self.tracked_objects[track_id]['counted_directions']) > 0

      thickness = 3 if counted_any else 2
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    color, thickness)

      # Draw label
      label = f"{class_name} #{track_id}"
      if counted_any:
        label += " ✓"
        # Show which directions
        directions = list(self.tracked_objects[track_id]['counted_directions'])
        if directions:
          label += f" [{directions[0][:3]}]"
      else:
        label += f" ({detection_count}/{self.min_detections})"

      # Label background
      (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(frame, (int(x1), int(y1) - lh - 10),
                    (int(x1) + lw, int(y1)), color, -1)
      cv2.putText(frame, label, (int(x1), int(y1) - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

      # Draw center point
      cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

    # Update detection history for non-detected objects
    current_track_ids = set(tid for tid, _ in tracked_detections)
    for track_id in list(self.tracked_objects.keys()):
      if track_id not in current_track_ids:
        self.tracked_objects[track_id]['detection_history'].append(False)

    # Draw zones and lines
    self._draw_zones_and_lines(frame)

    # Draw statistics
    # self._draw_stats(frame)

    self.frame_count += 1
    return frame

  def _draw_zones_and_lines(self, frame):
    """Draw detection zones and counting lines"""
    # Draw zones with transparency
    overlay = frame.copy()
    for zone in self.detection_zones:
      cv2.polylines(frame, [zone], True, (0, 255, 0), 2)
      cv2.fillPoly(overlay, [zone], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw counting lines
    for line_data in self.counting_lines:
      pts = line_data['points']
      direction = line_data['direction']

      # Draw line
      cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 3)

      # Draw endpoints
      cv2.circle(frame, tuple(pts[0]), 6, (255, 0, 0), -1)
      cv2.circle(frame, tuple(pts[1]), 6, (255, 0, 0), -1)

      # Draw direction label
      mid_x = (pts[0][0] + pts[1][0]) // 2
      mid_y = (pts[0][1] + pts[1][1]) // 2

      # Label background
      label = f"{direction}: {self.total_counts[direction]}"
      (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
      cv2.rectangle(frame, (mid_x - 5, mid_y - lh - 5),
                    (mid_x + lw + 5, mid_y + 5), (0, 0, 0), -1)
      cv2.putText(frame, label, (mid_x, mid_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

  def _draw_stats(self, frame):
    """Draw statistics panel"""
    h, w = frame.shape[:2]

    # Calculate panel size based on number of directions
    num_directions = len(self.counting_lines)
    panel_height = min(200 + num_directions * 30, h - 20)
    panel_width = 380

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - 10, 10),
                  (w - 10, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    x_pos = w - panel_width
    y_pos = 40

    # Title
    cv2.putText(frame, "DIRECTIONAL COUNTS", (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 35

    # Frame counter
    cv2.putText(frame, f"Frame: {self.frame_count}", (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 25

    # Overall total
    total_all = sum(self.total_counts.values())
    cv2.putText(frame, f"Total All: {total_all}", (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_pos += 30

    # Per-direction counts
    for line_data in self.counting_lines:
      direction = line_data['direction']
      count = self.total_counts[direction]

      cv2.putText(frame, f"{direction}: {count}", (x_pos, y_pos),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      y_pos += 25

    # Instructions
    y_pos += 10
    cv2.putText(frame, "Press 'q' to stop", (x_pos, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

  def process_video(self, video_path, output_path=None, show_preview=True):
    """Process video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f} seconds\n")

    writer = None
    if output_path:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
      print(f"Writing output to: {output_path}\n")

    print("Processing video... (Press 'q' to stop)\n")

    try:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break

        # Process frame
        annotated = self.process_frame(frame)

        # Write to output
        if writer:
          writer.write(annotated)

        # Show preview
        if show_preview:
          cv2.imshow('Directional Counter - Press Q to stop', annotated)

          key = cv2.waitKey(1) & 0xFF
          if key == ord('q'):
            print("\n\nStopped by user")
            break

        # Progress
        if self.frame_count % 30 == 0 or not show_preview:
          progress = (self.frame_count / total_frames) * 100
          total = sum(self.total_counts.values())
          print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames}) - "
                f"Total: {total}", end='\r')

    finally:
      cap.release()
      if writer:
        writer.release()
      if show_preview:
        cv2.destroyAllWindows()

    print(f"\n\nProcessing complete!")

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


def main():
  parser = argparse.ArgumentParser(
    description='Directional vehicle counter with zone support')

  now = datetime.now()

  # Model selection
  parser.add_argument('--model', '-m', type=str,
                      choices=['large', 'medium', 'small', 'nano_640', 'nano_512', 'nano_448', 'pico_512', 'femto_512', 'femto_352'],
                      default='large',
                      help='Model size')

  # Input/output
  parser.add_argument('--video', '-v', type=str, required=True,
                      help='Input video file')
  parser.add_argument('--output', '-o', type=str,
                      help='Output video file')

  # Zone configuration
  parser.add_argument('--config-zones', action='store_true',
                      help='Configure zones interactively')
  parser.add_argument('--zone-file', type=str,
                      help='Load zones from file')
  parser.add_argument('--save-zones', type=str,
                      help='Save zone configuration to file')

  # Detection parameters
  parser.add_argument('--conf', '-c', type=float, default=0.25,
                      help='Confidence threshold')
  parser.add_argument('--device', '-d', type=str, default='cpu',
                      help='Device: cuda:0 or cpu')

  # Counting parameters
  parser.add_argument('--min-detections', type=int, default=7,
                      help='Minimum detections to count')
  parser.add_argument('--detection-window', type=int, default=10,
                      help='Detection window size')

  # Display
  parser.add_argument('--no-preview', action='store_true',
                      help='Disable live preview')
  parser.add_argument('--stats', type=str,
                      help='Save statistics to JSON file')

  args = parser.parse_args()

  # Set model paths
  if args.model == 'large':
    config_path = 'configs/yolov8_l.py'
    checkpoint_path = 'weights/yolov8_l.pth'
    print("Using: YOLOv8-large MobileNetV2 (512x288)")
  elif args.model == 'medium':  # medium
    config_path = 'configs/yolov8_m.py'
    checkpoint_path = 'weights/yolov8_m.pth'
    print("Using: YOLOv8-medium (640x384)")
  elif args.model == 'small':
    config_path = 'configs/yolov8_s.py'
    checkpoint_path = 'weights/yolov8_s.pth'
  elif args.model == 'nano_640':
    config_path = 'configs/yolov8_n_640.py'
    checkpoint_path = 'weights/yolov8_n_640.pth'
  elif args.model == 'nano_512':
    config_path = 'configs/yolov8_n_512.py'
    checkpoint_path = 'weights/yolov8_n_512.pth'
  elif args.model == 'nano_448':
    config_path = 'configs/yolov8_n_448.py'
    checkpoint_path = 'weights/yolov8_n_448.pth'
  elif args.model == 'pico_512':
    config_path = 'configs/yolov8_p_512.py'
    checkpoint_path = 'weights/yolov8_p_512.pth'
  elif args.model == 'femto_512':
    config_path = 'configs/yolov8_f_512.py'
    checkpoint_path = 'weights/yolov8_f_512.pth'
  elif args.model == 'femto_352':
    config_path = 'configs/yolov8_f_352.py'
    checkpoint_path = 'weights/yolov8_f_352.pth'

  # Check files
  for filepath in [config_path, checkpoint_path, args.video]:
    if not Path(filepath).exists():
      print(f"ERROR: File not found: {filepath}")
      return

  # Get zone configuration
  zone_config = None

  if args.zone_file and Path(args.zone_file).exists():
    # Load from file
    print(f"Loading zone configuration from: {args.zone_file}")
    with open(args.zone_file, 'rb') as f:
      zone_config = pickle.load(f)
    print(f"✓ Loaded {len(zone_config['zones'])} zones and "
          f"{len(zone_config['counting_lines'])} counting lines")

  if args.config_zones or zone_config is None:
    # Interactive configuration
    print("Opening video to configure zones...")
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
      print("ERROR: Could not read video frame")
      return

    tool = ZoneConfigTool(frame)
    zone_config = tool.configure()

    # Save if requested
    if args.save_zones:
      with open(args.save_zones, 'wb') as f:
        pickle.dump(zone_config, f)
      print(f"✓ Zone configuration saved to: {args.save_zones}")

  if not zone_config or not zone_config['zones'] or not zone_config['counting_lines']:
    print("ERROR: Need at least one detection zone and one counting line")
    print("Run with --config-zones to set them up")
    return

  # Initialize counter
  counter = DirectionalCounter(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    zone_config=zone_config,
    conf_threshold=args.conf,
    device=args.device,
    min_detections=args.min_detections,
    detection_window=args.detection_window
  )

  # Process video
  stats = counter.process_video(
    video_path=args.video,
    output_path=args.output,
    show_preview=not args.no_preview
  )

  # Print results
  print("\n" + "=" * 70)
  print("FINAL STATISTICS")
  print("=" * 70)
  print(f"Total Vehicles (All Directions): {stats['total_all']}")
  print(f"Frames Processed: {stats['frames_processed']}")
  print(f"Time Spent: {datetime.now() - now}")

  print(f"\nCounts by Direction:")
  for direction, total in sorted(stats['directional_totals'].items()):
    print(f"\n  {direction}: {total} vehicles")

    if direction in stats['directional_breakdown']:
      breakdown = stats['directional_breakdown'][direction]
      for vehicle_class, count in sorted(breakdown.items()):
        if count > 0:
          pct = (count / total * 100) if total > 0 else 0
          print(f"    {vehicle_class:12s}: {count:4d} ({pct:5.1f}%)")

  print("=" * 70)

  # Save statistics
  if args.stats:
    with open(args.stats, 'w') as f:
      json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {args.stats}")


if __name__ == '__main__':
  main()

#  python3 directional_counter.py --video data/day.mp4 --zone-file intersection_config.pkl --output counted.mp4 --stats results.json --min-detections 15 --detection-window 20 --conf 0.3 --model femto_512 --no-preview
#  python3 directional_counter.py --video data/day.mp4 --config-zones --save-zones intersection_config.pkl