"""
Main Application Entry Point for Hand Gesture Recognition System.
Coordinates Camera Capture, AI Gesture Recognition, Virtual Air Canvas,
Virtual Mouse / Presentation / Media System Control, Real-Time Telemetry Analytics,
and Futuristic HUD Visualization.
"""
import sys
import time
import os
import cv2

import config
from gesture_detector import GestureDetector
from system_controller import SystemController
from analytics_engine import AnalyticsEngine
from ui_renderer import UIRenderer
from air_canvas import AirCanvas
from generate_analytics_report import generate_analytics_dashboard

def main():
    print("=" * 70)
    print("                HAND GESTURE RECOGNITION SYSTEM")
    print("=" * 70)
    print(f"[*] Initializing camera (Device Index: {config.CAMERA_INDEX})...")

    # Initialize Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[!] Primary camera index 0 failed. Trying fallback index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Could not open video capture device. Please ensure webcam is connected and accessible.")
        return

    # Set requested resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Camera initialized: {actual_w}x{actual_h} resolution.")

    # Initialize Engine Components
    print("[*] Loading Gesture Recognition Models...")
    detector = GestureDetector()
    controller = SystemController()
    analytics = AnalyticsEngine()
    renderer = UIRenderer()
    canvas = AirCanvas(actual_w, actual_h)

    current_mode_idx = 0
    current_mode = config.AVAILABLE_MODES[current_mode_idx]

    print("[*] Hand Gesture Recognition System is READY!")
    print("[*] Shortcuts: [M] Mode | [C] Mouse | [P] Snapshot | [E] Clear Canvas | [S] Export | [R] Reset | [H] Help | [Q] Quit")
    renderer.add_notification("System Ready - Press [H] for Gesture Guide", duration=3.0, color=config.COLOR_SECONDARY)

    window_name = "Hand Gesture Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    try:
        while True:
            start_frame_time = time.time()

            success, frame = cap.read()
            if not success or frame is None:
                print("[!] Frame read failed or stream ended.")
                break

            # Flip horizontally for intuitive mirror view
            frame = cv2.flip(frame, 1)

            # 1. Process Frame with AI Gesture Engine
            detected_hands = detector.process_frame(frame)
            primary_hand = None
            if detected_hands:
                primary_hand = next((h for h in detected_hands if h.hand_label == "Right"), detected_hands[0])

            # 2. Air Canvas Mode Processing
            if current_mode == config.MODE_AIR_CANVAS:
                frame, canvas_action = canvas.process(frame, primary_hand)
                if canvas_action:
                    renderer.add_notification(canvas_action, duration=1.0, color=config.COLOR_PRIMARY)

            # 3. System & Media Controller (Mouse, Scroll, Volume, Zoom, Slides, Media)
            if detected_hands and current_mode != config.MODE_AIR_CANVAS:
                action_executed = controller.update(primary_hand, actual_w, actual_h, current_mode)
                if action_executed:
                    renderer.add_notification(f"Action: {action_executed}", duration=1.2, color=config.COLOR_ACCENT)

            # 4. Update Analytics Engine
            proc_duration = time.time() - start_frame_time
            analytics.update_frame_telemetry(proc_duration)
            analytics.process_hands_data(detected_hands)

            # 5. Render Glowing Skeletons & Landmarks
            for hand in detected_hands:
                renderer.draw_hand_skeleton(frame, hand)

            # 6. Render HUD Analytics Cards & Telemetry
            renderer.draw_hud(frame, detected_hands, analytics, current_mode, controller.is_enabled)

            # 7. Display Frame
            cv2.imshow(window_name, frame)

            # 8. Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                renderer.add_notification("Saving session data and exiting...", duration=2.0)
                break

            elif key == ord('m') or key == ord('M'):
                # Cycle operating mode
                current_mode_idx = (current_mode_idx + 1) % len(config.AVAILABLE_MODES)
                current_mode = config.AVAILABLE_MODES[current_mode_idx]
                if current_mode == config.MODE_MOUSE_CONTROL:
                    controller.is_enabled = True
                else:
                    controller.is_enabled = False
                renderer.add_notification(f"Mode: {current_mode}", duration=1.8, color=config.COLOR_PRIMARY)

            elif key == ord('c') or key == ord('C'):
                # Toggle mouse control
                state = controller.toggle_control()
                status_text = "Virtual Mouse ENABLED" if state else "Virtual Mouse DISABLED"
                col = config.COLOR_ACCENT if state else config.COLOR_ALERT
                renderer.add_notification(status_text, duration=1.8, color=col)

            elif key == ord('p') or key == ord('P'):
                # Instant Snapshot / Photo Capture
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(config.SNAPSHOTS_DIR, f"snapshot_{timestamp}.png")
                cv2.imwrite(snap_path, frame)
                renderer.add_notification(f"Snapshot Captured!", duration=2.0, color=config.COLOR_ACCENT)
                print(f"[Snapshot] Saved to: {snap_path}")

            elif key == ord('e') or key == ord('E'):
                # Clear Air Canvas
                canvas.clear()
                renderer.add_notification("Air Canvas Cleared", duration=1.5, color=config.COLOR_PURPLE)

            elif key == ord('s') or key == ord('S'):
                # Export CSV & Generate Analytics Report Dashboard
                csv_path = analytics.export_csv()
                json_path = analytics.export_summary_json()
                report_path = generate_analytics_dashboard(csv_path)
                renderer.add_notification("Analytics Dashboard & CSV Exported!", duration=2.5, color=config.COLOR_SECONDARY)
                print(f"[Export] CSV: {csv_path}")
                print(f"[Export] Summary JSON: {json_path}")
                if report_path:
                    print(f"[Export] Dashboard Chart: {report_path}")

            elif key == ord('r') or key == ord('R'):
                # Reset Session Metrics
                analytics.reset_session()
                renderer.add_notification("Session Metrics Reset", duration=1.5, color=config.COLOR_PURPLE)

            elif key == ord('h') or key == ord('H'):
                # Toggle Help Guide Modal
                renderer.show_help = not renderer.show_help

    except KeyboardInterrupt:
        print("[*] Keyboard interrupt received.")
    finally:
        # Clean Shutdown & Automatic Report Generation
        print("[*] Releasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()

        print("[*] Finalizing Session Data Analytics...")
        csv_path = analytics.export_csv()
        json_path = analytics.export_summary_json()
        report_path = generate_analytics_dashboard(csv_path)

        print("=" * 70)
        print("  SESSION COMPLETED SUCCESSFULLY!")
        print(f"  - Total Frames: {analytics.total_frames}")
        print(f"  - Duration: {analytics.get_session_duration_str()}")
        print(f"  - Telemetry CSV: {csv_path}")
        print(f"  - Analytics Summary: {json_path}")
        if report_path:
            print(f"  - Visual Report Dashboard: {report_path}")
        print("=" * 70)

if __name__ == "__main__":
    main()
