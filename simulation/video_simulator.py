"""
wildnode_ai/simulation/video_simulator.py
==========================================
PURPOSE: Simulate a real-time camera/video detection stream using
         pre-recorded video frames or generating dummy frames.

HOW IT WORKS:
  - Generates or loads video frames
  - Runs vision detection on each frame
  - Yields results for the main pipeline
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_vision_simulation(interval: float = 4.0, iterations: int = None):
    """
    Generator that yields simulated vision detection results.

    Args:
        interval   : Seconds between frames
        iterations : Max number of frames (None = run forever)

    Yields:
        detections : List of detection dicts from detect_wildlife()
    """
    from vision_detection.detector import simulate_vision_detection

    count = 0
    print(f"[Vision Simulator] Starting camera stream (interval={interval}s)...")

    while True:
        detections = simulate_vision_detection()
        count += 1

        if detections:
            for d in detections:
                print(f"[Vision #{count:04d}] 📷 {d['label']}  Priority: {d['priority']}")
        else:
            print(f"[Vision #{count:04d}] 📷 Clear frame – no wildlife detected")

        yield {"frame_id": count, "detections": detections}

        if iterations is not None and count >= iterations:
            break

        time.sleep(interval)


if __name__ == "__main__":
    for result in run_vision_simulation(interval=2.0, iterations=5):
        pass
    print("\n[✓] Vision simulation complete.")
