"""
UI Renderer module for Hand Gesture Recognition System.
Renders futuristic HUD overlays, glassmorphic telemetry cards, glowing skeletal landmarks,
interactive status chips, touchless virtual buttons, and real-time mini-charts on OpenCV frames.
"""
import time
import math
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

import config
from gesture_detector import HandData
from analytics_engine import AnalyticsEngine

class UIRenderer:
    """
    Renders high-aesthetic HUD overlays, skeletons, and telemetry cards on OpenCV frames.
    """
    def __init__(self):
        self.show_help: bool = False
        self.notifications: List[Tuple[str, float, Tuple[int, int, int]]] = [] # (text, expiry_time, color)
        
        # Hand Skeleton connection lines (21 landmarks)
        self.CONNECTIONS = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm Base
            (5, 9), (9, 13), (13, 17)
        ]

    def add_notification(self, text: str, duration: float = 2.0, color: Tuple[int, int, int] = config.COLOR_ACCENT):
        """Adds a temporary on-screen toast notification."""
        self.notifications.append((text, time.time() + duration, color))

    @staticmethod
    def draw_transparent_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], alpha: float = 0.65):
        """Draws a semi-transparent rectangle card on the image."""
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # Subtle card border
        cv2.rectangle(img, (x, y), (x + w, y + h), (min(255, color[0] + 40), min(255, color[1] + 40), min(255, color[2] + 40)), 1)

    def draw_hand_skeleton(self, frame: np.ndarray, hand: HandData):
        """
        Renders futuristic glowing skeleton landmarks and directional vectors for a hand.
        """
        if len(hand.landmarks_px) < 21:
            return

        is_right = (hand.hand_label == "Right")
        joint_color = config.COLOR_CYAN if is_right else config.COLOR_SECONDARY
        line_color = (255, 180, 50) if is_right else (50, 180, 255)

        # Draw Connection Lines
        for p1_idx, p2_idx in self.CONNECTIONS:
            pt1 = hand.landmarks_px[p1_idx]
            pt2 = hand.landmarks_px[p2_idx]
            cv2.line(frame, pt1, pt2, (40, 40, 40), 4, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, line_color, 2, cv2.LINE_AA)

        # Draw Joint Landmarks
        for idx, pt in enumerate(hand.landmarks_px):
            # Fingertips have special colored outer rings
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pt, 8, config.COLOR_ACCENT, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 10, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw Palm Center & Velocity Vector
        cx, cy = hand.palm_center
        cv2.circle(frame, (cx, cy), 6, config.COLOR_ALERT, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, config.COLOR_ALERT, 1, cv2.LINE_AA)

        # Draw Motion Vector Arrow if moving fast
        if hand.speed > 100:
            vx_norm = hand.velocity_x / max(1.0, hand.speed)
            vy_norm = hand.velocity_y / max(1.0, hand.speed)
            arrow_len = min(60, int(hand.speed * 0.05))
            end_pt = (int(cx + vx_norm * arrow_len), int(cy + vy_norm * arrow_len))
            cv2.arrowedLine(frame, (cx, cy), end_pt, config.COLOR_ALERT, 2, tipLength=0.3)

        # Floating Gesture Pill Badge above hand
        x_min, y_min, x_max, y_max = hand.bbox
        pill_text = f"{hand.hand_label}: {hand.gesture_name} ({int(hand.gesture_confidence*100)}%)"
        (t_w, t_h), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        pill_x = max(10, x_min)
        pill_y = max(35, y_min - 15)

        self.draw_transparent_rect(frame, pill_x - 5, pill_y - t_h - 6, t_w + 14, t_h + 12, config.COLOR_CARD_BG, 0.85)
        cv2.rectangle(frame, (pill_x - 5, pill_y - t_h - 6), (pill_x + t_w + 9, pill_y + 6), joint_color, 1)
        cv2.putText(frame, pill_text, (pill_x + 2, pill_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)

    def draw_hud(self, frame: np.ndarray, hands: List[HandData], analytics: AnalyticsEngine, current_mode: str, mouse_enabled: bool):
        """
        Renders the complete modern HUD on top of the frame.
        """
        h, w, _ = frame.shape
        now = time.time()

        # -------------------------------------------------------------
        # 1. Top Header Bar (Title, Mode, FPS, Session Clock)
        # -------------------------------------------------------------
        self.draw_transparent_rect(frame, 15, 12, w - 30, 48, config.COLOR_BACKGROUND, 0.85)
        
        # Clean App Title (Updated: HAND GESTURE RECOGNITION)
        cv2.putText(frame, "HAND GESTURE RECOGNITION", (30, 42),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, config.COLOR_SECONDARY, 2, cv2.LINE_AA)

        # Active Mode Chip
        mode_colors = {
            config.MODE_GESTURE_RECOGNITION: config.COLOR_PRIMARY,
            config.MODE_AIR_CANVAS: (255, 50, 200),
            config.MODE_MOUSE_CONTROL: config.COLOR_ACCENT,
            config.MODE_FINGER_COUNTER: config.COLOR_PURPLE,
            config.MODE_PRESENTATION: config.COLOR_ALERT,
            config.MODE_MEDIA_CONTROL: config.COLOR_CYAN
        }
        mode_col = mode_colors.get(current_mode, config.COLOR_PRIMARY)
        mode_str = f"MODE: {current_mode}"
        (m_w, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        mode_box_x = w - 470
        self.draw_transparent_rect(frame, mode_box_x, 20, m_w + 18, 32, mode_col, 0.7)
        cv2.putText(frame, mode_str, (mode_box_x + 9, 41), cv2.FONT_HERSHEY_DUPLEX, 0.5, config.COLOR_TEXT, 1, cv2.LINE_AA)

        # Mouse Status Badge
        mouse_status_str = "MOUSE: ON" if mouse_enabled else "MOUSE: OFF"
        mouse_bg = config.COLOR_ACCENT if mouse_enabled else (70, 70, 80)
        self.draw_transparent_rect(frame, w - 280, 20, 110, 32, mouse_bg, 0.7)
        cv2.putText(frame, mouse_status_str, (w - 272, 41), cv2.FONT_HERSHEY_DUPLEX, 0.45, config.COLOR_TEXT, 1, cv2.LINE_AA)

        # Telemetry: FPS & Clock
        clock_str = f"{analytics.get_session_duration_str()} | {analytics.fps:.1f} FPS"
        cv2.putText(frame, clock_str, (w - 155, 42), cv2.FONT_HERSHEY_DUPLEX, 0.48, config.COLOR_TEXT, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. Left Panel: Live Hand & Gesture Analytics Card
        # -------------------------------------------------------------
        card_w = 290
        card_h = 320
        self.draw_transparent_rect(frame, 15, 70, card_w, card_h, config.COLOR_BACKGROUND, 0.85)

        cv2.putText(frame, "REAL-TIME TELEMETRY", (30, 95), cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_SECONDARY, 1, cv2.LINE_AA)
        cv2.line(frame, (30, 105), (15 + card_w - 20, 105), (80, 80, 95), 1)

        y_offset = 130
        if hands:
            for idx, hand in enumerate(hands):
                h_col = config.COLOR_CYAN if hand.hand_label == "Right" else config.COLOR_SECONDARY
                cv2.putText(frame, f"{hand.hand_label} Hand:", (30, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.5, h_col, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{hand.gesture_name}", (140, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.52, config.COLOR_TEXT, 1, cv2.LINE_AA)
                
                # Confidence Meter Bar
                y_offset += 16
                bar_w = 230
                conf_w = int(bar_w * hand.gesture_confidence)
                cv2.rectangle(frame, (30, y_offset), (30 + bar_w, y_offset + 6), (50, 50, 60), -1)
                cv2.rectangle(frame, (30, y_offset), (30 + conf_w, y_offset + 6), config.COLOR_ACCENT, -1)
                
                # Finger Count & Speed
                y_offset += 25
                cv2.putText(frame, f"Fingers: {hand.finger_count}/5", (30, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.46, config.COLOR_TEXT, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Speed: {int(hand.speed)} px/s", (150, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.46, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

                # Individual Finger Status Chips [T] [I] [M] [R] [P]
                y_offset += 25
                finger_short = [("T", "Thumb"), ("I", "Index"), ("M", "Middle"), ("R", "Ring"), ("P", "Pinky")]
                chip_x = 30
                for code, full_name in finger_short:
                    is_active = hand.finger_states.get(full_name, False)
                    chip_bg = config.COLOR_ACCENT if is_active else (60, 60, 70)
                    chip_fg = (0, 0, 0) if is_active else (160, 160, 160)
                    cv2.rectangle(frame, (chip_x, y_offset - 14), (chip_x + 36, y_offset + 4), chip_bg, -1)
                    cv2.putText(frame, code, (chip_x + 12, y_offset), cv2.FONT_HERSHEY_DUPLEX, 0.42, chip_fg, 1, cv2.LINE_AA)
                    chip_x += 44

                y_offset += 32
                if idx < len(hands) - 1:
                    cv2.line(frame, (30, y_offset - 10), (15 + card_w - 20, y_offset - 10), (60, 60, 70), 1)

            if len(hands) >= 2:
                total_fingers = sum(h.finger_count for h in hands)
                cv2.putText(frame, f"Combined Total Fingers: {total_fingers} / 10", (30, 365),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, config.COLOR_PURPLE, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Hands Detected", (50, 180), cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(frame, "Show hand to camera", (50, 210), cv2.FONT_HERSHEY_DUPLEX, 0.45, (100, 100, 110), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. Right Panel: Live Session Analytics Card
        # -------------------------------------------------------------
        r_card_w = 320
        r_card_h = 320
        r_card_x = w - r_card_w - 15
        self.draw_transparent_rect(frame, r_card_x, 70, r_card_w, r_card_h, config.COLOR_BACKGROUND, 0.85)

        cv2.putText(frame, "SESSION DATA ANALYTICS", (r_card_x + 15, 95), cv2.FONT_HERSHEY_DUPLEX, 0.55, config.COLOR_SECONDARY, 1, cv2.LINE_AA)
        cv2.line(frame, (r_card_x + 15, 105), (w - 30, 105), (80, 80, 95), 1)

        # Top Gestures Mini-Bar Chart
        top_gestures = analytics.get_top_gestures(4)
        gy_offset = 128
        cv2.putText(frame, "Top Gestures Occurrences:", (r_card_x + 15, gy_offset), cv2.FONT_HERSHEY_DUPLEX, 0.46, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        gy_offset += 20

        max_count = max([c for _, c, _ in top_gestures], default=1) or 1
        for g_name, g_count, g_pct in top_gestures:
            display_name = (g_name[:14] + '..') if len(g_name) > 16 else g_name
            cv2.putText(frame, display_name, (r_card_x + 15, gy_offset + 10), cv2.FONT_HERSHEY_DUPLEX, 0.42, config.COLOR_TEXT, 1, cv2.LINE_AA)
            b_start_x = r_card_x + 140
            b_max_w = 110
            bar_len = int((g_count / max_count) * b_max_w)
            cv2.rectangle(frame, (b_start_x, gy_offset), (b_start_x + b_max_w, gy_offset + 10), (50, 50, 60), -1)
            cv2.rectangle(frame, (b_start_x, gy_offset), (b_start_x + bar_len, gy_offset + 10), config.COLOR_PRIMARY, -1)
            cv2.putText(frame, f"{g_count}", (b_start_x + b_max_w + 8, gy_offset + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            gy_offset += 22

        gy_offset += 10
        r_pct, l_pct = analytics.get_hand_dominance_ratio()
        cv2.putText(frame, f"Hand Dominance: R: {r_pct}% | L: {l_pct}%", (r_card_x + 15, gy_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.44, config.COLOR_SECONDARY, 1, cv2.LINE_AA)

        gy_offset += 24
        cv2.putText(frame, "Recent Event Stream:", (r_card_x + 15, gy_offset), cv2.FONT_HERSHEY_DUPLEX, 0.44, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        gy_offset += 18
        
        events = list(analytics.recent_events)[:3]
        for ev in events:
            ev_str = f"[{ev['time']}] {ev['hand']}: {ev['gesture']} ({ev['fingers']}f)"
            cv2.putText(frame, ev_str, (r_card_x + 15, gy_offset), cv2.FONT_HERSHEY_DUPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
            gy_offset += 16

        # -------------------------------------------------------------
        # 4. Mode-Specific Visuals
        # -------------------------------------------------------------
        if current_mode == config.MODE_MOUSE_CONTROL and mouse_enabled:
            mx = int(w * 0.15)
            my = int(h * 0.15)
            cv2.rectangle(frame, (mx, my), (w - mx, h - my), config.COLOR_ACCENT, 1, cv2.LINE_AA)
            cv2.putText(frame, "Virtual Mouse Active Box", (mx + 10, my - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, config.COLOR_ACCENT, 1, cv2.LINE_AA)

        elif current_mode == config.MODE_FINGER_COUNTER and hands:
            total_f = sum(h.finger_count for h in hands)
            counter_text = f"TOTAL FINGERS: {total_f}"
            (cw, ch), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            cx = (w - cw) // 2
            self.draw_transparent_rect(frame, cx - 20, 80, cw + 40, ch + 30, config.COLOR_CARD_BG, 0.9)
            cv2.putText(frame, counter_text, (cx, 110), cv2.FONT_HERSHEY_DUPLEX, 1.2, config.COLOR_SECONDARY, 2, cv2.LINE_AA)

        elif current_mode == config.MODE_MEDIA_CONTROL:
            # Media control helper card in center
            m_card_w = 420
            m_card_h = 50
            mc_x = (w - m_card_w) // 2
            self.draw_transparent_rect(frame, mc_x, 70, m_card_w, m_card_h, config.COLOR_CARD_BG, 0.88)
            cv2.putText(frame, "MEDIA: [Palm] Play/Pause | [V] Vol+ | [ThumbsDn] Vol- | [3f] Mute",
                        (mc_x + 12, 102), cv2.FONT_HERSHEY_DUPLEX, 0.40, config.COLOR_CYAN, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 5. Bottom Navigation & Hotkeys Status Bar
        # -------------------------------------------------------------
        b_bar_h = 38
        self.draw_transparent_rect(frame, 15, h - b_bar_h - 12, w - 30, b_bar_h, config.COLOR_BACKGROUND, 0.85)
        
        shortcuts_text = "[M] Mode  |  [C] Mouse  |  [P] Snapshot  |  [E] Clear Canvas  |  [S] Export  |  [R] Reset  |  [H] Help  |  [Q] Exit"
        cv2.putText(frame, shortcuts_text, (30, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.44, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6. Toast Notifications System
        # -------------------------------------------------------------
        valid_notifs = []
        notif_y = h - 70
        for text, expiry, col in self.notifications:
            if now < expiry:
                valid_notifs.append((text, expiry, col))
                (nw, nh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
                nx = (w - nw) // 2
                self.draw_transparent_rect(frame, nx - 15, notif_y - nh - 8, nw + 30, nh + 16, config.COLOR_CARD_BG, 0.92)
                cv2.rectangle(frame, (nx - 15, notif_y - nh - 8), (nx + nw + 15, notif_y + 8), col, 2)
                cv2.putText(frame, text, (nx, notif_y), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 2, cv2.LINE_AA)
                notif_y -= 45
        self.notifications = valid_notifs

        # -------------------------------------------------------------
        # 7. Help Overlay Modal (Toggleable via [H])
        # -------------------------------------------------------------
        if self.show_help:
            self._render_help_modal(frame, w, h)

    def _render_help_modal(self, frame: np.ndarray, w: int, h: int):
        """Renders comprehensive help & gesture guide overlay modal."""
        modal_w = 680
        modal_h = 470
        mx = (w - modal_w) // 2
        my = (h - modal_h) // 2

        self.draw_transparent_rect(frame, mx, my, modal_w, modal_h, (15, 15, 20), 0.96)
        cv2.rectangle(frame, (mx, my), (mx + modal_w, my + modal_h), config.COLOR_PRIMARY, 2)

        cv2.putText(frame, "HAND GESTURE RECOGNITION GUIDE & SHORTCUTS", (mx + 30, my + 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, config.COLOR_SECONDARY, 2, cv2.LINE_AA)
        cv2.line(frame, (mx + 30, my + 50), (mx + modal_w - 30, my + 50), (100, 100, 120), 1)

        lines = [
            ("Thumbs Up [UN] / Like", "Smooth Scroll Up (in Mouse Mode) / Positive Gesture"),
            ("Thumbs Down [DON] / Dislike", "Smooth Scroll Down (in Mouse Mode) / Negative Gesture"),
            ("Index Pointing", "Moves Virtual Mouse Cursor / Draws in Air Canvas Mode"),
            ("Peace / Victory (2 Fingers)", "Volume Up (+) / Air Canvas Selection & Palette Hover"),
            ("Closed Fist (0 Fingers)", "Volume Down (-) / Air Canvas Fist Eraser Stamp"),
            ("Open Palm (5 Fingers)", "Media Play-Pause / 5 Fingers Count / Neutral"),
            ("Pinch (Thumb + Index)", "Triggers Virtual Mouse Click / Select"),
            ("OK Sign", "Zoom In (Ctrl + +)"),
            ("Swipe Left / Right", "Previous / Next Presentation Slide or Track"),
            ("", ""),
            ("KEYBOARD HOTKEYS & FEATURES:", ""),
            ("[M] Key", "Cycle Modes: RECOGNITION -> AIR_CANVAS -> MOUSE -> COUNTER -> SLIDES -> MEDIA"),
            ("[C] Key", "Toggle Virtual Mouse Control ON / OFF"),
            ("[P] Key", "Capture Instant High-Quality Snapshot to session_data/snapshots/"),
            ("[E] Key", "Clear Air Canvas Drawing"),
            ("[S] Key", "Export CSV Telemetry & Generate 6-Panel Analytics Dashboard"),
            ("[R] Key", "Reset Current Session Counters"),
            ("[H] Key", "Toggle this Help Guide ON / OFF"),
            ("[Q] / [ESC] Key", "Save Session & Exit Application")
        ]

        ly = my + 75
        for col1, col2 in lines:
            if not col1 and not col2:
                ly += 6
                continue
            if col1.startswith("KEYBOARD"):
                cv2.putText(frame, col1, (mx + 30, ly), cv2.FONT_HERSHEY_DUPLEX, 0.48, config.COLOR_CYAN, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"- {col1}:", (mx + 30, ly), cv2.FONT_HERSHEY_DUPLEX, 0.42, config.COLOR_TEXT, 1, cv2.LINE_AA)
                cv2.putText(frame, col2, (mx + 260, ly), cv2.FONT_HERSHEY_DUPLEX, 0.42, config.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            ly += 21
