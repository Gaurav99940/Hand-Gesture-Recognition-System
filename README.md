# 🖐️ Hand Gesture Recognition System

An advanced **AI-powered Hand Gesture Recognition & Interactive Vision System** built using Python, MediaPipe Vision Tasks, OpenCV, PyAutoGUI, Pandas, and Matplotlib.

This system accesses your webcam, accurately tracks hand movements in real-time, counts fingers, recognizes directional gestures (Thumbs Up / Down, Victory, Pointing, OK, Fist, Swipes), provides live **Telemetry HUD overlays**, enables **Virtual Air Canvas (Drawing in the air)**, **Virtual Mouse & Media Controls**, and automatically generates **Analytics Dashboards (CSV + Visual Charts)**.

---

## 🚀 Key Features

1. **Real-Time Hand Tracking & Landmark Geometry**:
   - 21 3D Hand Landmarks with neon skeletal rendering.
   - Dual-Hand support (Left & Right hands simultaneously).
   - High-precision Finger Angle & Distance Vector computations.

2. **🎨 Virtual Air Canvas (Gesture Painting)** *(NEW!)*:
   - Draw glowing neon lines in the air using your **Index Finger**!
   - Switch brush colors (Cyan, Green, Yellow, Red, Purple, White, Eraser) using on-screen virtual palette buttons.
   - Hover and select colors effortlessly with the **Victory / 2 Fingers (✌️)** gesture.
   - Clear canvas with the **`[E]`** key or virtual `[CLEAR]` button.
   - Fist acts as a live eraser stamp.

3. **🎵 Media Player Controller** *(NEW!)*:
   - Control YouTube / Spotify / VLC hands-free:
     - **Open Palm (🖐️) / Fist (✊)**: Play / Pause.
     - **Victory (✌️)**: Volume Up (+).
     - **Thumbs Down (👎)**: Volume Down (-).
     - **3 Fingers / Call Me (🤙)**: Mute / Unmute.
     - **Swipe Left / Right**: Previous / Next Track.

4. **📸 Instant Snapshot Capture** *(NEW!)*:
   - Press **`[P]`** to capture an instant high-resolution frame with HUD annotations, saved automatically into `session_data/snapshots/`.

5. **Accurate Finger Counting & State Badges**:
   - Real-time status for every individual finger: **Thumb, Index, Middle, Ring, Pinky** (`EXTENDED` vs `FOLDED`).
   - Counts extended fingers from **0 to 5** per hand (and **0 to 10** combined).
   - Live visual indicator chips on screen: `[T] [I] [M] [R] [P]`.

6. **Rich Gesture Recognition Engine**:
   - 👍 **Thumbs Up / UN**: Positive gesture / Smooth Scroll Up.
   - 👎 **Thumbs Down / DON**: Negative gesture / Smooth Scroll Down.
   - ☝️ **Pointing (Up/Down/Left/Right)**: Directional navigation & Cursor movement.
   - ✌️ **Peace / Victory (2 Fingers)**: Volume Up (+).
   - ✊ **Closed Fist (0 Fingers)**: Volume Down (-) / Stop / Eraser.
   - 🖐️ **Open Palm (5 Fingers)**: Play-Pause / Neutral / Reset.
   - 👌 **OK Sign**: Zoom In (`Ctrl + +`).
   - 🤏 **Pinch (Thumb + Index)**: Virtual Mouse Click / Select.
   - 💨 **Motion Swipes (Left / Right / Up / Down)**: Rapid hand movement detection for presentation slides and tracks.

7. **Data Analytics & Telemetry Engine**:
   - **Live HUD Overlays**: Real-time FPS, Latency (ms), Gesture Confidence %, Hand Speed (px/s), Motion Direction.
   - **Live Distribution Bar Charts**: On-screen mini-chart tracking top gestures in the active session.
   - **Hand Dominance Ratio**: Real-time balance of Right Hand vs. Left Hand interactions.
   - **Automated Report Generation**: Exports clean CSV datasets & 6-panel graphical dashboards upon session completion or on keypress `[S]`.

8. **Multi-Mode Operation**:
   - 🔍 **RECOGNITION MODE** (Default): Gesture detection, finger counting, and telemetry HUD.
   - 🎨 **AIR CANVAS MODE**: Finger painting, color palette, neon brush strokes.
   - 🖱️ **MOUSE CONTROL MODE**: Virtual mouse cursor with exponential smoothing, click, scroll, and volume hotkeys.
   - 🔢 **FINGER COUNTER MODE**: Large high-visibility finger count display & math breakdown.
   - 📽️ **PRESENTATION MODE**: Hands-free slide navigation (Next/Prev slide via swipes or pointing).
   - 🎵 **MEDIA CONTROL MODE**: Media playback control (Play/Pause, Mute, Track navigation).

---

## 📂 Project Architecture

```
Hand Gesture Recognition System/
│
├── main.py                     # Master Application Entry Point
├── air_canvas.py               # Virtual Air Canvas & Finger Painting Engine
├── gesture_detector.py         # AI Gesture Recognition & Finger Geometry Engine
├── analytics_engine.py         # Real-time Telemetry, Data Logging & Metrics Tracker
├── ui_renderer.py              # Futuristic Glassmorphic HUD & Skeleton Visualizer
├── system_controller.py        # Virtual Mouse, Scroll, Volume, Media & Presentation Controller
├── generate_analytics_report.py# 6-Panel Analytics Dashboard Chart Generator
├── config.py                   # Centralized System Settings & Color Palette
├── requirements.txt            # Python Dependencies
├── models/                     # MediaPipe Neural Network Task Models (.task)
└── session_data/               # Output Directory for CSV Logs, Snapshots & Dashboard PNGs
    └── snapshots/              # Saved high-res snapshots and air canvas drawings
```

---

## 🎮 Keyboard Hotkeys & Controls

| Key | Action | Description |
|---|---|---|
| `[M]` | **Cycle Mode** | Switch between `RECOGNITION` ➔ `AIR_CANVAS` ➔ `MOUSE_CONTROL` ➔ `FINGER_COUNTER` ➔ `PRESENTATION` ➔ `MEDIA_CONTROL` |
| `[C]` | **Toggle Mouse** | Enable / Disable virtual mouse control safely |
| `[P]` | **Snapshot** | Capture and save an instant photo to `session_data/snapshots/` |
| `[E]` | **Clear Canvas** | Clear all drawings in Air Canvas Mode |
| `[S]` | **Export Data** | Immediately export CSV telemetry log and generate 6-panel Analytics Dashboard PNG |
| `[R]` | **Reset Session** | Reset all counters and metrics for a new session |
| `[H]` | **Help Guide** | Toggle the on-screen gesture & hotkey guide modal |
| `[Q]` / `[ESC]` | **Exit** | Save all session data, generate final analytics report, and exit |

---

## 🛠️ How to Run the Project

1. Open PowerShell or Command Prompt in the project folder:
   ```powershell
   cd "e:\Hand Gesture Recognition System"
   ```

2. Run using the virtual environment:
   ```powershell
   & ".\.venv\Scripts\python.exe" main.py
   ```
   *(Or run `python hi.py`)*

3. To view or generate an analytics dashboard for previous sessions:
   ```powershell
   & ".\.venv\Scripts\python.exe" generate_analytics_report.py
   ```
