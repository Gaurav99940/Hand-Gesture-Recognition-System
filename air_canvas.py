"""
Air Canvas / Gesture Drawing Engine for Hand Gesture Recognition System.
Allows users to draw in the air using finger gestures with multiple neon colors,
brush sizes, eraser, and on-screen interactive color palette.
"""
import os
import time
from typing import List, Tuple, Optional
import cv2
import numpy as np

import config
from gesture_detector import HandData

class AirCanvas:
    """
    Virtual Air Canvas for drawing with hand gestures in real time.
    """
    def __init__(self, width: int = config.CAMERA_WIDTH, height: int = config.CAMERA_HEIGHT):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Current drawing state
        self.current_color_idx: int = 0
        self.colors = config.CANVAS_PALETTE
        self.brush_thickness: int = 6
        self.eraser_thickness: int = 40
        
        # Previous point for line interpolation
        self.prev_pt: Optional[Tuple[int, int]] = None
        self.is_drawing: bool = False
        
        # Palette button layout at top of screen
        self.palette_buttons: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        self._init_palette_buttons()

    def _init_palette_buttons(self):
        """Initializes on-screen virtual color buttons."""
        btn_w = 90
        btn_h = 42
        start_x = 320
        start_y = 68
        gap = 12

        self.palette_buttons = []
        for idx, (name, color) in enumerate(self.colors):
            bx = start_x + idx * (btn_w + gap)
            by = start_y
            self.palette_buttons.append((name, color, (bx, by, bx + btn_w, by + btn_h)))

        # Add [CLEAR] Button
        bx = start_x + len(self.colors) * (btn_w + gap) + 15
        self.palette_buttons.append(("CLEAR", (50, 50, 200), (bx, start_y, bx + btn_w + 10, start_y + btn_h)))

    def clear(self):
        """Clears the entire drawing canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_pt = None
        print("[AirCanvas] Canvas cleared.")

    def save_drawing(self) -> str:
        """Saves current canvas drawing to snapshot directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(config.SNAPSHOTS_DIR, f"drawing_{timestamp}.png")
        cv2.imwrite(filepath, self.canvas)
        print(f"[AirCanvas] Drawing saved to: {filepath}")
        return filepath

    def process(self, frame: np.ndarray, hand: Optional[HandData]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Updates air canvas based on hand position and gestures:
        - Index Finger ONLY: Drawing Mode (Draws glowing neon lines)
        - Index + Middle Fingers (Victory / 2 Fingers): Selection Mode (Move brush without drawing, click palette buttons)
        - Closed Fist: Eraser Stamp
        """
        action_notification = None

        # Check canvas dimensions match frame
        fh, fw, _ = frame.shape
        if fh != self.height or fw != self.width:
            self.height, self.width = fh, fw
            self.canvas = cv2.resize(self.canvas, (fw, fh))
            self._init_palette_buttons()

        # Render Palette Buttons Bar at top
        self._draw_palette_bar(frame)

        if not hand or len(hand.landmarks_px) < 21:
            self.prev_pt = None
            self.is_drawing = False
            return self._blend_canvas(frame), None

        index_tip = hand.landmarks_px[8]
        middle_tip = hand.landmarks_px[12]
        thumb_tip = hand.landmarks_px[4]

        index_up = hand.finger_states.get("Index", False)
        middle_up = hand.finger_states.get("Middle", False)
        ring_up = hand.finger_states.get("Ring", False)
        pinky_up = hand.finger_states.get("Pinky", False)
        thumb_up = hand.finger_states.get("Thumb", False)

        curr_color_name, curr_color = self.colors[self.current_color_idx]
        is_eraser = (curr_color_name == "Eraser")

        # -------------------------------------------------------------
        # 1. Selection & Hover Mode: Index & Middle Fingers Up (✌️)
        # -------------------------------------------------------------
        if index_up and middle_up and not ring_up and not pinky_up:
            self.prev_pt = None
            self.is_drawing = False

            # Draw visual selection crosshair between fingertips
            select_pt = ((index_tip[0] + middle_tip[0]) // 2, (index_tip[1] + middle_tip[1]) // 2)
            cv2.circle(frame, select_pt, 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, select_pt, 6, config.COLOR_CYAN, -1, cv2.LINE_AA)
            cv2.putText(frame, "Selecting / Hovering", (select_pt[0] + 16, select_pt[1] + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)

            # Check if touching any palette buttons
            sx, sy = select_pt
            for idx, (name, col, (bx1, by1, bx2, by2)) in enumerate(self.palette_buttons):
                if bx1 <= sx <= bx2 and by1 <= sy <= by2:
                    if name == "CLEAR":
                        self.clear()
                        action_notification = "Canvas Cleared!"
                    else:
                        self.current_color_idx = idx
                        action_notification = f"Selected: {name}"

        # -------------------------------------------------------------
        # 2. Drawing Mode: Index Finger ONLY (☝️)
        # -------------------------------------------------------------
        elif index_up and not middle_up and not ring_up and not pinky_up:
            draw_pt = index_tip
            self.is_drawing = True

            # Brush cursor on camera frame
            brush_rad = self.eraser_thickness // 2 if is_eraser else self.brush_thickness + 2
            cv2.circle(frame, draw_pt, brush_rad, (0, 0, 0) if is_eraser else curr_color, -1, cv2.LINE_AA)
            cv2.circle(frame, draw_pt, brush_rad + 3, (255, 255, 255), 1, cv2.LINE_AA)

            if self.prev_pt is not None:
                # Interpolate line on canvas
                if is_eraser:
                    cv2.line(self.canvas, self.prev_pt, draw_pt, (0, 0, 0), self.eraser_thickness, cv2.LINE_AA)
                else:
                    # Neon Glow effect: thicker transparent base + intense core
                    cv2.line(self.canvas, self.prev_pt, draw_pt, curr_color, self.brush_thickness + 4, cv2.LINE_AA)
                    cv2.line(self.canvas, self.prev_pt, draw_pt, (255, 255, 255), max(1, self.brush_thickness - 2), cv2.LINE_AA)

            self.prev_pt = draw_pt

        # -------------------------------------------------------------
        # 3. Fist Mode: Eraser Stamp
        # -------------------------------------------------------------
        elif hand.gesture_name == config.GESTURE_FIST:
            cx, cy = hand.palm_center
            cv2.circle(self.canvas, (cx, cy), 45, (0, 0, 0), -1)
            cv2.circle(frame, (cx, cy), 45, (100, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(frame, "Fist Eraser", (cx - 40, cy - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, config.COLOR_ALERT, 1, cv2.LINE_AA)
            self.prev_pt = None
            self.is_drawing = False

        else:
            self.prev_pt = None
            self.is_drawing = False

        return self._blend_canvas(frame), action_notification

    def _draw_palette_bar(self, frame: np.ndarray):
        """Renders interactive color palette bar at the top."""
        for idx, (name, col, (bx1, by1, bx2, by2)) in enumerate(self.palette_buttons):
            is_active = (idx == self.current_color_idx) and (name != "CLEAR")
            
            # Button background
            btn_bg = col if name != "CLEAR" and name != "Eraser" else (40, 40, 50)
            if is_active:
                cv2.rectangle(frame, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), (255, 255, 255), 3)
            
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_bg, -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)

            # Button text
            text_col = (0, 0, 0) if (name in ["Yellow", "Cyan", "White", "Green"]) else (255, 255, 255)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            tx = bx1 + (bx2 - bx1 - tw) // 2
            ty = by1 + (by2 - by1 + th) // 2
            cv2.putText(frame, name, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.42, text_col, 1, cv2.LINE_AA)

    def _blend_canvas(self, frame: np.ndarray) -> np.ndarray:
        """Blends canvas strokes seamlessly onto video frame with alpha blending."""
        # Convert canvas to grayscale mask
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black out the drawing area in the camera frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        # Take drawing from canvas
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        # Combine
        combined = cv2.add(frame_bg, canvas_fg)
        return combined
