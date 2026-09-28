"""
System Controller module for Virtual Mouse, Volume, Scroll, Zoom, Media, and Presentation Control.
Uses PyAutoGUI with smoothing filters and debounce protection.
"""
import time
import math
from typing import Tuple, Optional
import pyautogui

import config
from gesture_detector import HandData

# Configure PyAutoGUI safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

class SystemController:
    """
    Translates recognized hand gestures and landmarks into smooth OS system commands.
    """
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.is_enabled: bool = False  # Toggleable to prevent accidental mouse hijacking
        
        # Smoothed cursor position
        self.prev_cursor_x: float = self.screen_w / 2
        self.prev_cursor_y: float = self.screen_h / 2
        self.smoothing: float = config.MOUSE_SMOOTHING
        
        # Action debounce timers
        self.last_action_time: float = 0.0
        self.last_scroll_time: float = 0.0
        self.last_click_time: float = 0.0
        self.is_clicking: bool = False

    def toggle_control(self) -> bool:
        """Toggles the system controller ON or OFF."""
        self.is_enabled = not self.is_enabled
        return self.is_enabled

    def update(self, hand: HandData, frame_w: int, frame_h: int, current_mode: str = config.MODE_GESTURE_RECOGNITION) -> Optional[str]:
        """
        Processes the active hand and executes corresponding system actions based on mode.
        Returns the action name if an action was executed, else None.
        """
        now = time.time()
        executed_action = None

        if not hand or len(hand.landmarks_px) < 21:
            return None

        # Handle Dedicated Media Control Mode
        if current_mode == config.MODE_MEDIA_CONTROL:
            return self._handle_media_mode(hand, now)

        # In Mouse Control mode, or when mouse is enabled
        if self.is_enabled or current_mode == config.MODE_MOUSE_CONTROL:
            executed_action = self._handle_mouse_and_gestures(hand, frame_w, frame_h, now)

        # In Presentation Mode
        elif current_mode == config.MODE_PRESENTATION:
            if now - self.last_action_time > config.GESTURE_COOLDOWN:
                if hand.swipe_direction == config.GESTURE_SWIPE_RIGHT or hand.gesture_name == config.GESTURE_POINT_RIGHT:
                    try:
                        pyautogui.press("right")
                        self.last_action_time = now
                        executed_action = "Next Slide ->"
                    except Exception:
                        pass
                elif hand.swipe_direction == config.GESTURE_SWIPE_LEFT or hand.gesture_name == config.GESTURE_POINT_LEFT:
                    try:
                        pyautogui.press("left")
                        self.last_action_time = now
                        executed_action = "<- Prev Slide"
                    except Exception:
                        pass

        return executed_action

    def _handle_media_mode(self, hand: HandData, now: float) -> Optional[str]:
        """Handles Media Player controls (Play/Pause, Volume, Mute, Next/Prev)."""
        if now - self.last_action_time <= config.GESTURE_COOLDOWN:
            return None

        executed = None
        # 1. Play / Pause (Open Palm or Fist toggle)
        if hand.gesture_name == config.GESTURE_OPEN_PALM or hand.gesture_name == config.GESTURE_FIST:
            try:
                pyautogui.press("playpause")
                self.last_action_time = now
                executed = "Media Play / Pause"
            except Exception:
                pass

        # 2. Volume Up (Victory ✌️)
        elif hand.gesture_name == config.GESTURE_VICTORY:
            try:
                pyautogui.press("volumeup")
                self.last_action_time = now
                executed = "Volume Up (+)"
            except Exception:
                pass

        # 3. Volume Down (Thumbs Down 👎)
        elif hand.gesture_name == config.GESTURE_THUMBS_DOWN:
            try:
                pyautogui.press("volumedown")
                self.last_action_time = now
                executed = "Volume Down (-)"
            except Exception:
                pass

        # 4. Mute / Unmute (Call Me / 3 Fingers 🤙)
        elif hand.gesture_name == config.GESTURE_CALL_ME or hand.finger_count == 3:
            try:
                pyautogui.press("volumemute")
                self.last_action_time = now
                executed = "Mute / Unmute"
            except Exception:
                pass

        # 5. Next Track / Forward (Swipe Right or Point Right)
        elif hand.swipe_direction == config.GESTURE_SWIPE_RIGHT or hand.gesture_name == config.GESTURE_POINT_RIGHT:
            try:
                pyautogui.press("nexttrack")
                self.last_action_time = now
                executed = "Next Track >>"
            except Exception:
                pass

        # 6. Prev Track / Rewind (Swipe Left or Point Left)
        elif hand.swipe_direction == config.GESTURE_SWIPE_LEFT or hand.gesture_name == config.GESTURE_POINT_LEFT:
            try:
                pyautogui.press("prevtrack")
                self.last_action_time = now
                executed = "<< Prev Track"
            except Exception:
                pass

        return executed

    def _handle_mouse_and_gestures(self, hand: HandData, frame_w: int, frame_h: int, now: float) -> Optional[str]:
        """Handles Virtual Mouse movement, clicking, scrolling, and zoom."""
        executed_action = None
        index_tip_px = hand.landmarks_px[8]

        # 1. Virtual Mouse Movement
        margin_x = int(frame_w * 0.15)
        margin_y = int(frame_h * 0.15)
        active_w = frame_w - 2 * margin_x
        active_h = frame_h - 2 * margin_y

        rel_x = (index_tip_px[0] - margin_x) / max(1, active_w)
        rel_y = (index_tip_px[1] - margin_y) / max(1, active_h)

        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        target_x = rel_x * self.screen_w
        target_y = rel_y * self.screen_h

        curr_x = self.prev_cursor_x + (target_x - self.prev_cursor_x) / self.smoothing
        curr_y = self.prev_cursor_y + (target_y - self.prev_cursor_y) / self.smoothing

        if hand.finger_states.get("Index", False):
            try:
                pyautogui.moveTo(int(curr_x), int(curr_y))
                self.prev_cursor_x = curr_x
                self.prev_cursor_y = curr_y
            except Exception:
                pass

        # 2. Click Gesture (Pinch between Index Tip and Thumb Tip)
        pinch_dist = math.hypot(
            hand.landmarks_norm[8][0] - hand.landmarks_norm[4][0],
            hand.landmarks_norm[8][1] - hand.landmarks_norm[4][1]
        )

        if pinch_dist < config.CLICK_DISTANCE_THRESH:
            if not self.is_clicking and (now - self.last_click_time > 0.35):
                try:
                    pyautogui.click()
                    self.is_clicking = True
                    self.last_click_time = now
                    executed_action = "Mouse Click"
                except Exception:
                    pass
        else:
            self.is_clicking = False

        # 3. Gesture Actions (Scroll, Volume, Presentation, Zoom)
        if now - self.last_action_time > config.GESTURE_COOLDOWN:
            if hand.gesture_name == config.GESTURE_THUMBS_UP:
                try:
                    pyautogui.scroll(config.SCROLL_SPEED)
                    self.last_action_time = now
                    executed_action = "Scroll Up"
                except Exception:
                    pass

            elif hand.gesture_name == config.GESTURE_THUMBS_DOWN:
                try:
                    pyautogui.scroll(-config.SCROLL_SPEED)
                    self.last_action_time = now
                    executed_action = "Scroll Down"
                except Exception:
                    pass

            elif hand.gesture_name == config.GESTURE_VICTORY:
                try:
                    pyautogui.press("volumeup")
                    self.last_action_time = now
                    executed_action = "Volume Up (+)"
                except Exception:
                    pass

            elif hand.gesture_name == config.GESTURE_FIST:
                try:
                    pyautogui.press("volumedown")
                    self.last_action_time = now
                    executed_action = "Volume Down (-)"
                except Exception:
                    pass

            elif hand.gesture_name == config.GESTURE_OK:
                try:
                    pyautogui.hotkey("ctrl", "+")
                    self.last_action_time = now
                    executed_action = "Zoom In"
                except Exception:
                    pass

            elif hand.swipe_direction == config.GESTURE_SWIPE_RIGHT or hand.gesture_name == config.GESTURE_POINT_RIGHT:
                try:
                    pyautogui.press("right")
                    self.last_action_time = now
                    executed_action = "Next Slide ->"
                except Exception:
                    pass

            elif hand.swipe_direction == config.GESTURE_SWIPE_LEFT or hand.gesture_name == config.GESTURE_POINT_LEFT:
                try:
                    pyautogui.press("left")
                    self.last_action_time = now
                    executed_action = "<- Prev Slide"
                except Exception:
                    pass

        return executed_action
