"""
Configuration module for Hand Gesture Recognition System.
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SESSION_DATA_DIR = os.path.join(BASE_DIR, "session_data")
SNAPSHOTS_DIR = os.path.join(SESSION_DATA_DIR, "snapshots")

# Ensure required directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SESSION_DATA_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# MediaPipe Model Paths
GESTURE_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_recognizer.task")
HAND_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# Camera Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Detection & Tracking Thresholds
MIN_DETECTION_CONFIDENCE = 0.65
MIN_PRESENCE_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.65
NUM_HANDS = 2

# Mouse & System Control Configuration
MOUSE_SMOOTHING = 5          # Smoothing factor for cursor movement (EMA / Moving Average)
CLICK_DISTANCE_THRESH = 0.045  # Normalized distance between thumb tip & index tip for click
PINCH_HOLD_TIME = 0.3        # Seconds to confirm a pinch click
GESTURE_COOLDOWN = 0.6       # Minimum seconds between system trigger actions (volume, scroll, etc.)
SCROLL_SPEED = 50            # Units to scroll per step

# Motion / Velocity Thresholds
SWIPE_VELOCITY_THRESH = 800   # Velocity (pixels/second) to detect swipe
SWIPE_COOLDOWN = 0.7         # Seconds cooldown after swipe

# Operating Modes
MODE_GESTURE_RECOGNITION = "RECOGNITION"
MODE_AIR_CANVAS = "AIR_CANVAS"
MODE_MOUSE_CONTROL = "MOUSE_CONTROL"
MODE_FINGER_COUNTER = "FINGER_COUNTER"
MODE_PRESENTATION = "PRESENTATION"
MODE_MEDIA_CONTROL = "MEDIA_CONTROL"

AVAILABLE_MODES = [
    MODE_GESTURE_RECOGNITION,
    MODE_AIR_CANVAS,
    MODE_MOUSE_CONTROL,
    MODE_FINGER_COUNTER,
    MODE_PRESENTATION,
    MODE_MEDIA_CONTROL
]

# Color Palette (BGR for OpenCV)
COLOR_PRIMARY = (255, 140, 0)      # Vivid Blue/Cyan
COLOR_SECONDARY = (0, 215, 255)    # Gold/Yellow
COLOR_ACCENT = (50, 205, 50)       # Lime Green
COLOR_BACKGROUND = (20, 20, 25)    # Dark Slate
COLOR_CARD_BG = (35, 35, 45)       # Card Slate
COLOR_TEXT = (255, 255, 255)       # White
COLOR_TEXT_DIM = (180, 180, 180)   # Light Gray
COLOR_ALERT = (50, 50, 255)        # Red/Orange
COLOR_PURPLE = (220, 20, 180)      # Purple Accent
COLOR_CYAN = (240, 240, 50)        # Bright Cyan
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)

# Air Canvas Palette Choices
CANVAS_PALETTE = [
    ("Cyan", (255, 255, 0)),
    ("Green", (50, 255, 50)),
    ("Yellow", (0, 215, 255)),
    ("Red", (50, 50, 255)),
    ("Purple", (220, 20, 180)),
    ("White", (255, 255, 255)),
    ("Eraser", (0, 0, 0))
]

# Standard Gesture Names
GESTURE_THUMBS_UP = "Thumbs Up"
GESTURE_THUMBS_DOWN = "Thumbs Down"
GESTURE_VICTORY = "Peace / Victory"
GESTURE_FIST = "Closed Fist"
GESTURE_OPEN_PALM = "Open Palm"
GESTURE_POINT_UP = "Pointing Up"
GESTURE_POINT_DOWN = "Pointing Down"
GESTURE_POINT_LEFT = "Pointing Left"
GESTURE_POINT_RIGHT = "Pointing Right"
GESTURE_OK = "OK Sign"
GESTURE_ROCK = "Rock On / Horns"
GESTURE_CALL_ME = "Call Me"
GESTURE_PINCH = "Pinch"
GESTURE_SWIPE_LEFT = "Swipe Left"
GESTURE_SWIPE_RIGHT = "Swipe Right"
GESTURE_SWIPE_UP = "Swipe Up"
GESTURE_SWIPE_DOWN = "Swipe Down"
GESTURE_GUN = "Gun / Snap"
GESTURE_UNKNOWN = "Unknown / Moving"
