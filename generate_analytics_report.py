"""
Analytics Report & Visualization Generator for Hand Gesture Recognition System.
Generates publication-quality charts and dashboards from recorded session datasets.
"""
import os
import glob
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/background rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

import config

def generate_analytics_dashboard(csv_filepath: str, output_image_path: str = None) -> str:
    """
    Reads a session CSV telemetry file and generates a multi-panel visual analytics report.
    """
    if not os.path.exists(csv_filepath):
        print(f"[ReportGenerator] File not found: {csv_filepath}")
        return ""

    df = pd.read_csv(csv_filepath)
    if df.empty or "gesture" not in df.columns:
        print("[ReportGenerator] CSV contains no valid gesture telemetry.")
        return ""

    if not output_image_path:
        base_name = os.path.splitext(os.path.basename(csv_filepath))[0]
        output_image_path = os.path.join(config.SESSION_DATA_DIR, f"report_{base_name}.png")

    # Set modern dark-mode style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 11), dpi=120)
    fig.patch.set_facecolor('#141419')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    # 1. Main Title
    session_time = df['session_time_s'].max() if 'session_time_s' in df.columns else 0
    total_samples = len(df)
    fig.suptitle(
        f"HAND GESTURE RECOGNITION DASHBOARD\n"
        f"Session Duration: {int(session_time)}s | Total Samples: {total_samples:,}",
        fontsize=16, fontweight='bold', color='#00D4FF', y=0.98
    )

    # -------------------------------------------------------------
    # Panel 1: Gesture Frequency Distribution (Bar Chart)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1E1E26')
    gesture_counts = df['gesture'].value_counts()

    if not gesture_counts.empty:
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(gesture_counts)))
        bars = ax1.barh(gesture_counts.index[::-1], gesture_counts.values[::-1], color=colors, edgecolor='#00D4FF', linewidth=0.8)
        ax1.set_title("Gesture Frequency Distribution", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
        ax1.set_xlabel("Occurrences (Frames)", fontsize=10, color='#AAAAAA')
        ax1.grid(axis='x', linestyle='--', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + max(gesture_counts.values)*0.02, bar.get_y() + bar.get_height()/2,
                     f"{int(width)}", ha='left', va='center', color='#FFFFFF', fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No Gesture Data", ha='center', va='center', color='#888888')

    # -------------------------------------------------------------
    # Panel 2: Spatial Movement & Heatmap Density (Hand Trajectory)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1E1E26')
    if 'palm_x' in df.columns and 'palm_y' in df.columns:
        valid_coords = df.dropna(subset=['palm_x', 'palm_y'])
        if not valid_coords.empty:
            scatter = ax2.scatter(
                valid_coords['palm_x'], valid_coords['palm_y'],
                c=valid_coords.index, cmap='plasma', alpha=0.6, s=15, edgecolors='none'
            )
            ax2.set_xlim(0, config.CAMERA_WIDTH)
            ax2.set_ylim(config.CAMERA_HEIGHT, 0) # Invert Y for screen coords
            ax2.set_title("Hand Spatial Trajectory & Motion Path", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
            ax2.set_xlabel("Camera X (px)", fontsize=10, color='#AAAAAA')
            ax2.set_ylabel("Camera Y (px)", fontsize=10, color='#AAAAAA')
            cbar = plt.colorbar(scatter, ax=ax2, orientation='horizontal', pad=0.18, aspect=30)
            cbar.set_label('Timeline Progression (Early -> Late)', color='#AAAAAA', fontsize=8)
            cbar.ax.tick_params(labelsize=8)
    else:
        ax2.text(0.5, 0.5, "No Spatial Data", ha='center', va='center', color='#888888')

    # -------------------------------------------------------------
    # Panel 3: Left vs Right Hand Dominance (Donut Chart)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#1E1E26')
    if 'hand' in df.columns:
        hand_counts = df['hand'].value_counts()
        if not hand_counts.empty:
            pie_colors = ['#FF8C00', '#00D4FF'] if len(hand_counts) == 2 else ['#00D4FF']
            wedges, texts, autotexts = ax3.pie(
                hand_counts.values, labels=hand_counts.index, autopct='%1.1f%%',
                startangle=140, colors=pie_colors,
                wedgeprops=dict(width=0.45, edgecolor='#141419', linewidth=2),
                textprops=dict(color='#FFFFFF', fontsize=10)
            )
            for at in autotexts:
                at.set_color('#141419')
                at.set_fontweight('bold')
            ax3.set_title("Hand Dominance Ratio", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
    else:
        ax3.text(0.5, 0.5, "No Hand Data", ha='center', va='center', color='#888888')

    # -------------------------------------------------------------
    # Panel 4: Individual Finger Usage Rates (%)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('#1E1E26')
    finger_cols = ['thumb_up', 'index_up', 'middle_up', 'ring_up', 'pinky_up']
    finger_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    if all(col in df.columns for col in finger_cols):
        finger_pcts = [df[col].mean() * 100 for col in finger_cols]
        bars = ax4.bar(finger_labels, finger_pcts, color='#32CD32', edgecolor='#FFFFFF', linewidth=0.5, width=0.55)
        ax4.set_title("Finger Extension Activity Rate (%)", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
        ax4.set_ylabel("Extension Rate %", fontsize=10, color='#AAAAAA')
        ax4.set_ylim(0, 105)
        ax4.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 2, f"{height:.1f}%",
                     ha='center', va='bottom', color='#FFFFFF', fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No Finger Data", ha='center', va='center', color='#888888')

    # -------------------------------------------------------------
    # Panel 5: Finger Count Distribution (0 to 5)
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor('#1E1E26')
    if 'finger_count' in df.columns:
        fc_counts = df['finger_count'].value_counts().sort_index()
        all_counts = [fc_counts.get(i, 0) for i in range(6)]
        bars = ax5.bar([f"{i} Fingers" for i in range(6)], all_counts, color='#BA55D3', edgecolor='#FFFFFF', linewidth=0.5, width=0.55)
        ax5.set_title("Finger Count Frequency (0 to 5)", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
        ax5.set_ylabel("Frames", fontsize=10, color='#AAAAAA')
        ax5.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, height + max(all_counts)*0.02, f"{int(height)}",
                         ha='center', va='bottom', color='#FFFFFF', fontsize=9)
    else:
        ax5.text(0.5, 0.5, "No Count Data", ha='center', va='center', color='#888888')

    # -------------------------------------------------------------
    # Panel 6: Hand Motion Velocity Profile Over Time (Speed)
    # -------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('#1E1E26')
    if 'session_time_s' in df.columns and 'speed' in df.columns:
        # Rolling mean for smooth visual speed line
        speed_smooth = df['speed'].rolling(window=7, min_periods=1).mean()
        ax6.plot(df['session_time_s'], speed_smooth, color='#FF4500', linewidth=1.5, label='Speed (px/s)')
        ax6.fill_between(df['session_time_s'], 0, speed_smooth, color='#FF4500', alpha=0.25)
        ax6.set_title("Hand Motion Velocity Profile", fontsize=12, fontweight='bold', color='#FFFFFF', pad=10)
        ax6.set_xlabel("Session Timeline (seconds)", fontsize=10, color='#AAAAAA')
        ax6.set_ylabel("Speed (pixels/s)", fontsize=10, color='#AAAAAA')
        ax6.grid(linestyle='--', alpha=0.3)
        max_speed = df['speed'].max()
        ax6.axhline(config.SWIPE_VELOCITY_THRESH, color='#00FFFF', linestyle=':', label='Swipe Threshold')
        ax6.legend(loc='upper right', fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No Speed Data", ha='center', va='center', color='#888888')

    # Save figure to file
    plt.savefig(output_image_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[ReportGenerator] High-Resolution Analytics Dashboard saved: {output_image_path}")
    return output_image_path


if __name__ == "__main__":
    # Test or standalone execution: find latest session CSV
    csv_files = glob.glob(os.path.join(config.SESSION_DATA_DIR, "*.csv"))
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"Generating analytics report for: {latest_csv}")
        generate_analytics_dashboard(latest_csv)
    else:
        print("No session CSV files found in session_data/. Run main.py first to record gesture telemetry.")
