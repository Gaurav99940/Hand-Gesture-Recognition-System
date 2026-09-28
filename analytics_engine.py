"""
Data Analytics Engine for Hand Gesture Recognition.
Tracks session metrics, gesture frequencies, finger distributions, spatial heatmaps,
velocity profiles, and exports structured CSV / JSON datasets.
"""
import os
import time
import json
import csv
from datetime import datetime
from collections import Counter, deque
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
import config
from gesture_detector import HandData

class AnalyticsEngine:
    """
    Real-Time Data Analytics & Telemetry Engine for Hand Gesture Tracking.
    """
    def __init__(self):
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: float = time.time()
        self.total_frames: int = 0
        self.hands_detected_frames: int = 0
        
        # FPS and Latency telemetry
        self.frame_times = deque(maxlen=30)
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        
        # Gesture Analytics
        self.gesture_counter: Counter = Counter()
        self.gesture_duration: Dict[str, float] = {}
        self.current_active_gesture: Dict[str, Tuple[str, float]] = {} # Hand -> (Gesture, StartTime)
        
        # Finger Analytics
        self.finger_usage_counter: Counter = Counter()
        self.finger_count_counter: Counter = Counter()
        
        # Handedness Analytics (Left vs Right)
        self.handedness_counter: Counter = Counter()
        
        # Motion & Spatial Analytics
        self.trajectory_points: List[Dict[str, Any]] = [] # For spatial heatmap & path plots
        self.total_distance_px: float = 0.0
        self.max_speed_px_s: float = 0.0
        self.prev_palm_pos: Dict[str, Tuple[int, int]] = {}
        
        # Live Event Log (Recent 20 events for HUD display)
        self.recent_events = deque(maxlen=20)
        
        # Comprehensive Data Log for CSV Export
        self.raw_logs: List[Dict[str, Any]] = []

    def update_frame_telemetry(self, process_duration: float):
        """Updates FPS and computation latency."""
        now = time.time()
        self.total_frames += 1
        self.frame_times.append(now)
        self.latency_ms = process_duration * 1000.0

        if len(self.frame_times) > 1:
            dt = self.frame_times[-1] - self.frame_times[0]
            if dt > 0:
                self.fps = (len(self.frame_times) - 1) / dt

    def process_hands_data(self, hands: List[HandData]):
        """
        Ingests real-time frame hand data and updates statistical distributions.
        """
        now = time.time()
        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if hands:
            self.hands_detected_frames += 1

        for hand in hands:
            hand_label = hand.hand_label
            gesture = hand.gesture_name
            finger_count = hand.finger_count
            conf = hand.gesture_confidence
            speed = hand.speed
            cx, cy = hand.palm_center

            # 1. Update Handedness Counter
            self.handedness_counter[hand_label] += 1

            # 2. Update Gesture Counter
            if gesture != config.GESTURE_UNKNOWN:
                self.gesture_counter[gesture] += 1

            # 3. Update Gesture Duration Tracking
            if hand_label in self.current_active_gesture:
                prev_gesture, g_start = self.current_active_gesture[hand_label]
                if prev_gesture == gesture:
                    self.gesture_duration[gesture] = self.gesture_duration.get(gesture, 0.0) + (now - g_start)
                    self.current_active_gesture[hand_label] = (gesture, now)
                else:
                    self.gesture_duration[prev_gesture] = self.gesture_duration.get(prev_gesture, 0.0) + (now - g_start)
                    self.current_active_gesture[hand_label] = (gesture, now)
                    # New gesture event triggered
                    self.recent_events.appendleft({
                        "time": timestamp_str,
                        "hand": hand_label,
                        "gesture": gesture,
                        "fingers": finger_count,
                        "conf": round(conf, 2)
                    })
            else:
                self.current_active_gesture[hand_label] = (gesture, now)
                self.recent_events.appendleft({
                    "time": timestamp_str,
                    "hand": hand_label,
                    "gesture": gesture,
                    "fingers": finger_count,
                    "conf": round(conf, 2)
                })

            # 4. Finger Analytics
            self.finger_count_counter[finger_count] += 1
            for finger_name, is_up in hand.finger_states.items():
                if is_up:
                    self.finger_usage_counter[finger_name] += 1

            # 5. Motion Dynamics & Distance
            if hand_label in self.prev_palm_pos:
                px_prev, py_prev = self.prev_palm_pos[hand_label]
                dist = ((cx - px_prev)**2 + (cy - py_prev)**2)**0.5
                self.total_distance_px += dist
            self.prev_palm_pos[hand_label] = (cx, cy)

            if speed > self.max_speed_px_s:
                self.max_speed_px_s = speed

            # Spatial point sample (sample 1 per 3 frames to keep memory light)
            if self.total_frames % 3 == 0:
                self.trajectory_points.append({
                    "time_offset_s": round(now - self.start_time, 2),
                    "hand": hand_label,
                    "x": cx,
                    "y": cy,
                    "speed": round(speed, 1),
                    "gesture": gesture
                })

            # 6. Granular Raw Log (Sampled for CSV export)
            if self.total_frames % 2 == 0:
                self.raw_logs.append({
                    "timestamp": timestamp_str,
                    "session_time_s": round(now - self.start_time, 3),
                    "frame_id": self.total_frames,
                    "hand": hand_label,
                    "gesture": gesture,
                    "confidence": round(conf, 3),
                    "finger_count": finger_count,
                    "thumb_up": hand.finger_states.get("Thumb", False),
                    "index_up": hand.finger_states.get("Index", False),
                    "middle_up": hand.finger_states.get("Middle", False),
                    "ring_up": hand.finger_states.get("Ring", False),
                    "pinky_up": hand.finger_states.get("Pinky", False),
                    "palm_x": cx,
                    "palm_y": cy,
                    "velocity_x": round(hand.velocity_x, 2),
                    "velocity_y": round(hand.velocity_y, 2),
                    "speed": round(speed, 2),
                    "swipe": hand.swipe_direction or "None"
                })

    def get_session_duration_str(self) -> str:
        """Returns formatted session elapsed time HH:MM:SS."""
        elapsed = int(time.time() - self.start_time)
        hrs = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def get_top_gestures(self, n: int = 5) -> List[Tuple[str, int, float]]:
        """
        Returns top N gestures with (name, count, percentage_of_total).
        """
        total = sum(self.gesture_counter.values()) or 1
        return [(g, c, round((c / total) * 100, 1)) for g, c in self.gesture_counter.most_common(n)]

    def get_hand_dominance_ratio(self) -> Tuple[float, float]:
        """Returns (Right Hand %, Left Hand %)."""
        total = sum(self.handedness_counter.values()) or 1
        right_pct = (self.handedness_counter.get("Right", 0) / total) * 100
        left_pct = (self.handedness_counter.get("Left", 0) / total) * 100
        return round(right_pct, 1), round(left_pct, 1)

    def export_csv(self, filename: Optional[str] = None) -> str:
        """
        Exports granular telemetry logs to CSV file for pandas/Excel analytics.
        """
        if not filename:
            filename = f"gesture_telemetry_{self.session_id}.csv"
        filepath = os.path.join(config.SESSION_DATA_DIR, filename)

        if not self.raw_logs:
            # Create a placeholder row if empty
            df = pd.DataFrame([{"session_id": self.session_id, "status": "No hands detected in session"}])
        else:
            df = pd.DataFrame(self.raw_logs)

        df.to_csv(filepath, index=False)
        print(f"[AnalyticsEngine] Telemetry exported to CSV: {filepath}")
        return filepath

    def export_summary_json(self, filename: Optional[str] = None) -> str:
        """
        Exports high-level analytical metrics summary to JSON.
        """
        if not filename:
            filename = f"session_summary_{self.session_id}.json"
        filepath = os.path.join(config.SESSION_DATA_DIR, filename)

        right_pct, left_pct = self.get_hand_dominance_ratio()

        summary = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(time.time() - self.start_time, 2),
            "total_frames_processed": self.total_frames,
            "hands_detected_frames": self.hands_detected_frames,
            "hand_detection_rate_pct": round((self.hands_detected_frames / max(1, self.total_frames)) * 100, 2),
            "average_fps": round(self.fps, 2),
            "hand_dominance": {
                "right_hand_pct": right_pct,
                "left_hand_pct": left_pct
            },
            "gesture_frequency": dict(self.gesture_counter),
            "finger_usage_frequency": dict(self.finger_usage_counter),
            "finger_count_frequency": dict(self.finger_count_counter),
            "total_hand_distance_px": round(self.total_distance_px, 1),
            "max_speed_px_s": round(self.max_speed_px_s, 1)
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print(f"[AnalyticsEngine] Session summary exported to JSON: {filepath}")
        return filepath

    def reset_session(self):
        """Resets all metrics for a new session."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.total_frames = 0
        self.hands_detected_frames = 0
        self.gesture_counter.clear()
        self.gesture_duration.clear()
        self.current_active_gesture.clear()
        self.finger_usage_counter.clear()
        self.finger_count_counter.clear()
        self.handedness_counter.clear()
        self.trajectory_points.clear()
        self.total_distance_px = 0.0
        self.max_speed_px_s = 0.0
        self.prev_palm_pos.clear()
        self.recent_events.clear()
        self.raw_logs.clear()
        print("[AnalyticsEngine] Session metrics have been reset.")
