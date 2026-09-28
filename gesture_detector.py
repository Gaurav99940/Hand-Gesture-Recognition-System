"""
Hand Gesture Detector module using MediaPipe Tasks & Landmark Geometry.
Provides high-accuracy finger counting, gesture classification, and motion tracking.
"""
import os
import time
import math
import urllib.request
from collections import deque
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

class HandData:
    """Encapsulates all computed data for a single detected hand."""
    def __init__(self):
        self.hand_label: str = "Right"       # "Right" or "Left"
        self.confidence: float = 0.0         # Detection confidence
        self.landmarks_norm: List[Tuple[float, float, float]] = []   # Normalized (0..1)
        self.landmarks_px: List[Tuple[int, int]] = []                # Pixel coords (x, y)
        self.bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)          # (x_min, y_min, x_max, y_max)
        self.palm_center: Tuple[int, int] = (0, 0)
        
        # Finger states: True if finger is extended, False if folded
        self.finger_states: Dict[str, bool] = {
            "Thumb": False,
            "Index": False,
            "Middle": False,
            "Ring": False,
            "Pinky": False
        }
        self.finger_count: int = 0           # Count of extended fingers (0-5)
        self.gesture_name: str = config.GESTURE_UNKNOWN
        self.gesture_confidence: float = 0.0
        
        # Motion metrics
        self.velocity_x: float = 0.0        # px/sec
        self.velocity_y: float = 0.0        # px/sec
        self.speed: float = 0.0             # px/sec
        self.swipe_direction: Optional[str] = None


class GestureDetector:
    """
    Robust Hand Landmark and Gesture Recognition Engine.
    Combines MediaPipe Deep Learning Task Recognizer with Geometric Heuristics.
    """
    def __init__(self, model_path: str = config.GESTURE_MODEL_PATH):
        self._ensure_model_exists(model_path)
        
        # Initialize MediaPipe Gesture Recognizer
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        
        # Motion tracking history: hand_label -> deque of (timestamp, x, y)
        self.trajectory_history: Dict[str, deque] = {
            "Right": deque(maxlen=15),
            "Left": deque(maxlen=15)
        }
        self.last_swipe_time: float = 0.0

    def _ensure_model_exists(self, model_path: str):
        """Downloads the MediaPipe gesture recognizer model if missing."""
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"[GestureDetector] Model not found at {model_path}. Downloading from Google Cloud...")
            url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            urllib.request.urlretrieve(url, model_path)
            print("[GestureDetector] Download completed successfully.")

    @staticmethod
    def _calc_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Euclidean distance in 3D normalized space."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    @staticmethod
    def _calc_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Calculate angle in degrees at vertex b formed by points a-b-c."""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba) * math.hypot(*bc) + 1e-7)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return math.degrees(math.acos(cosine_angle))

    def _determine_finger_states(self, landmarks_norm: List[Tuple[float, float, float]], hand_label: str) -> Dict[str, bool]:
        """
        Determines whether each of the 5 fingers is EXTENDED (True) or FOLDED (False)
        using multi-joint vector geometry and relative wrist distances.
        Landmarks index map:
          Thumb:  1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
          Index:  5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
          Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
          Ring:   13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
          Pinky:  17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)
          Wrist:  0
        """
        wrist = landmarks_norm[0]
        states = {}

        # 1. Thumb Extended Detection:
        # Distance from thumb tip (4) to pinky MCP (17) vs thumb IP (3) to pinky MCP (17)
        # and distance from thumb tip (4) to wrist (0)
        thumb_tip = landmarks_norm[4]
        thumb_ip = landmarks_norm[3]
        thumb_mcp = landmarks_norm[2]
        pinky_mcp = landmarks_norm[17]
        index_mcp = landmarks_norm[5]

        dist_tip_pinky = self._calc_distance(thumb_tip, pinky_mcp)
        dist_ip_pinky = self._calc_distance(thumb_ip, pinky_mcp)
        dist_tip_wrist = self._calc_distance(thumb_tip, wrist)
        dist_mcp_wrist = self._calc_distance(thumb_mcp, wrist)

        # Thumb angle check
        thumb_angle = self._calc_angle((thumb_mcp[0], thumb_mcp[1]), (thumb_ip[0], thumb_ip[1]), (thumb_tip[0], thumb_tip[1]))

        # Robust Thumb detection for mirrored camera view
        thumb_extended = (dist_tip_pinky > dist_ip_pinky * 1.08) and (dist_tip_wrist > dist_mcp_wrist * 1.1) and (thumb_angle > 140)
        states["Thumb"] = bool(thumb_extended)

        # 2. Four Non-Thumb Fingers (Index, Middle, Ring, Pinky)
        finger_indices = [
            ("Index", 5, 6, 7, 8),
            ("Middle", 9, 10, 11, 12),
            ("Ring", 13, 14, 15, 16),
            ("Pinky", 17, 18, 19, 20)
        ]

        for name, mcp_idx, pip_idx, dip_idx, tip_idx in finger_indices:
            tip = landmarks_norm[tip_idx]
            pip = landmarks_norm[pip_idx]
            mcp = landmarks_norm[mcp_idx]

            # Compare distances from wrist: if tip is significantly further from wrist than PIP/MCP
            dist_tip_wrist = self._calc_distance(tip, wrist)
            dist_pip_wrist = self._calc_distance(pip, wrist)
            dist_mcp_wrist = self._calc_distance(mcp, wrist)

            angle = self._calc_angle((mcp[0], mcp[1]), (pip[0], pip[1]), (tip[0], tip[1]))

            # Finger is extended if tip is further from wrist than pip and angle is straight (>145 deg)
            is_extended = (dist_tip_wrist > dist_pip_wrist * 1.05) and (dist_pip_wrist > dist_mcp_wrist) and (angle > 135)
            states[name] = bool(is_extended)

        return states

    def _classify_gesture(
        self,
        model_gesture: str,
        model_score: float,
        finger_states: Dict[str, bool],
        landmarks_norm: List[Tuple[float, float, float]],
        hand_label: str
    ) -> Tuple[str, float]:
        """
        Classifies the exact gesture by fusing MediaPipe model predictions
        with precise geometric rules.
        """
        thumb = finger_states["Thumb"]
        index = finger_states["Index"]
        middle = finger_states["Middle"]
        ring = finger_states["Ring"]
        pinky = finger_states["Pinky"]

        count = sum([thumb, index, middle, ring, pinky])
        wrist = landmarks_norm[0]
        thumb_tip = landmarks_norm[4]
        index_tip = landmarks_norm[8]
        middle_tip = landmarks_norm[12]
        pinky_tip = landmarks_norm[20]

        # 1. OK Sign: Thumb tip and Index tip touching/close, Middle/Ring/Pinky extended
        dist_thumb_index = self._calc_distance(thumb_tip, index_tip)
        if dist_thumb_index < 0.055 and (middle or ring or pinky):
            return config.GESTURE_OK, 0.95

        # 2. Thumbs Up / Thumbs Down:
        # Only Thumb extended (or other 4 folded)
        if not index and not middle and not ring and not pinky:
            # Check vertical orientation of thumb relative to wrist
            if thumb_tip[1] < wrist[1] - 0.08:
                return config.GESTURE_THUMBS_UP, 0.95
            elif thumb_tip[1] > wrist[1] + 0.08:
                return config.GESTURE_THUMBS_DOWN, 0.95
            elif thumb:
                if thumb_tip[1] < wrist[1]:
                    return config.GESTURE_THUMBS_UP, 0.90
                else:
                    return config.GESTURE_THUMBS_DOWN, 0.90
            else:
                return config.GESTURE_FIST, 0.92

        # 3. Closed Fist: 0 fingers extended
        if count == 0:
            return config.GESTURE_FIST, 0.95

        # 4. Pointing / Directional: Only Index extended (Thumb can be optionally out or folded)
        if index and not middle and not ring and not pinky:
            # Check direction of index finger vector (mcp -> tip)
            index_mcp = landmarks_norm[5]
            dx = index_tip[0] - index_mcp[0]
            dy = index_tip[1] - index_mcp[1]

            if abs(dy) > abs(dx):
                if dy < 0:
                    return config.GESTURE_POINT_UP, 0.94
                else:
                    return config.GESTURE_POINT_DOWN, 0.94
            else:
                if dx < 0:
                    return config.GESTURE_POINT_LEFT, 0.94
                else:
                    return config.GESTURE_POINT_RIGHT, 0.94

        # 5. Peace / Victory (✌️): Index & Middle extended, Ring & Pinky folded
        if index and middle and not ring and not pinky:
            return config.GESTURE_VICTORY, 0.96

        # 6. Rock On / Horns (🤘): Index & Pinky extended, Middle & Ring folded
        if index and pinky and not middle and not ring:
            return config.GESTURE_ROCK, 0.94

        # 7. Call Me (🤙): Thumb & Pinky extended, Index, Middle, Ring folded
        if thumb and pinky and not index and not middle and not ring:
            return config.GESTURE_CALL_ME, 0.93

        # 8. Open Palm (🖐️): 5 fingers extended
        if count == 5:
            return config.GESTURE_OPEN_PALM, 0.95

        # 9. Pinch: Thumb and Index close
        if dist_thumb_index < 0.045:
            return config.GESTURE_PINCH, 0.88

        # 10. Fallback to MediaPipe built-in model prediction if high confidence
        model_map = {
            "Thumb_Up": config.GESTURE_THUMBS_UP,
            "Thumb_Down": config.GESTURE_THUMBS_DOWN,
            "Victory": config.GESTURE_VICTORY,
            "Closed_Fist": config.GESTURE_FIST,
            "Open_Palm": config.GESTURE_OPEN_PALM,
            "Pointing_Up": config.GESTURE_POINT_UP,
            "ILoveYou": config.GESTURE_ROCK
        }
        if model_gesture in model_map and model_score > 0.6:
            return model_map[model_gesture], model_score

        # Finger count fallback
        return f"{count} Fingers", 0.85

    def _track_motion(self, hand_label: str, palm_center: Tuple[int, int]) -> Tuple[float, float, float, Optional[str]]:
        """
        Calculates hand velocity (px/sec) and detects rapid directional swipe gestures.
        """
        now = time.time()
        history = self.trajectory_history[hand_label]
        history.append((now, palm_center[0], palm_center[1]))

        if len(history) < 3:
            return 0.0, 0.0, 0.0, None

        # Compare oldest and newest in history window (~0.15s)
        t_old, x_old, y_old = history[0]
        t_new, x_new, y_new = history[-1]
        dt = t_new - t_old

        if dt < 0.04:
            return 0.0, 0.0, 0.0, None

        vx = (x_new - x_old) / dt
        vy = (y_new - y_old) / dt
        speed = math.hypot(vx, vy)

        swipe = None
        if speed > config.SWIPE_VELOCITY_THRESH and (now - self.last_swipe_time > config.SWIPE_COOLDOWN):
            if abs(vx) > abs(vy) * 1.3:
                if vx > 0:
                    swipe = config.GESTURE_SWIPE_RIGHT
                else:
                    swipe = config.GESTURE_SWIPE_LEFT
                self.last_swipe_time = now
            elif abs(vy) > abs(vx) * 1.3:
                if vy > 0:
                    swipe = config.GESTURE_SWIPE_DOWN
                else:
                    swipe = config.GESTURE_SWIPE_UP
                self.last_swipe_time = now

        return vx, vy, speed, swipe

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Processes an OpenCV BGR frame, detects hands, computes landmarks,
        finger counts, gesture classifications, and motion dynamics.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run MediaPipe Task Recognizer
        recognition_result = self.recognizer.recognize(mp_image)

        detected_hands: List[HandData] = []

        if not recognition_result.hand_landmarks:
            return detected_hands

        for idx, landmarks in enumerate(recognition_result.hand_landmarks):
            hand = HandData()

            # Handedness (Left / Right)
            if idx < len(recognition_result.handedness) and recognition_result.handedness[idx]:
                handedness_category = recognition_result.handedness[idx][0]
                # In mirrored camera, MediaPipe's "Left" hand corresponds to user's "Right" hand and vice versa
                hand.hand_label = handedness_category.category_name
                hand.confidence = handedness_category.score
            else:
                hand.hand_label = "Right" if idx == 0 else "Left"
                hand.confidence = 0.90

            # Store coordinates
            xs, ys = [], []
            hand.landmarks_norm = []
            hand.landmarks_px = []

            for lm in landmarks:
                hand.landmarks_norm.append((lm.x, lm.y, lm.z))
                px_x, px_y = int(lm.x * w), int(lm.y * h)
                hand.landmarks_px.append((px_x, px_y))
                xs.append(px_x)
                ys.append(px_y)

            # Bounding box
            x_min = max(0, min(xs) - 20)
            y_min = max(0, min(ys) - 20)
            x_max = min(w, max(xs) + 20)
            y_max = min(h, max(ys) + 20)
            hand.bbox = (x_min, y_min, x_max, y_max)

            # Palm center (average of wrist [0], index_mcp [5], and pinky_mcp [17])
            cx = int((hand.landmarks_px[0][0] + hand.landmarks_px[5][0] + hand.landmarks_px[17][0]) / 3)
            cy = int((hand.landmarks_px[0][1] + hand.landmarks_px[5][1] + hand.landmarks_px[17][1]) / 3)
            hand.palm_center = (cx, cy)

            # Finger states & count
            hand.finger_states = self._determine_finger_states(hand.landmarks_norm, hand.hand_label)
            hand.finger_count = sum(hand.finger_states.values())

            # Model gesture
            model_gesture_name = "None"
            model_gesture_score = 0.0
            if idx < len(recognition_result.gestures) and recognition_result.gestures[idx]:
                top_gesture = recognition_result.gestures[idx][0]
                model_gesture_name = top_gesture.category_name
                model_gesture_score = top_gesture.score

            # Combined gesture classification
            hand.gesture_name, hand.gesture_confidence = self._classify_gesture(
                model_gesture=model_gesture_name,
                model_score=model_gesture_score,
                finger_states=hand.finger_states,
                landmarks_norm=hand.landmarks_norm,
                hand_label=hand.hand_label
            )

            # Motion velocity & swipe tracking
            vx, vy, speed, swipe = self._track_motion(hand.hand_label, hand.palm_center)
            hand.velocity_x = vx
            hand.velocity_y = vy
            hand.speed = speed
            hand.swipe_direction = swipe

            detected_hands.append(hand)

        return detected_hands
