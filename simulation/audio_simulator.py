"""
wildnode_ai/simulation/audio_simulator.py
==========================================
PURPOSE: Simulate a real-time audio detection stream using pre-recorded
         clips. In production, this would be replaced by a live microphone
         feed from an edge device.

HOW IT WORKS:
  - Calls the audio predict module in a loop
  - Simulates delay between detections (interval)
  - Yields detection results for the main pipeline
"""

import os
import sys
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_audio_simulation(interval: float = 5.0, iterations: int = None):
    """
    Generator that yields simulated audio detection results.

    Args:
        interval   : Seconds between detections (simulates real-time feed)
        iterations : Max number of detections (None = run forever)

    Yields:
        result : Audio prediction dict
    """
    from audio_detection.predict import simulate_audio_detection

    count = 0
    print(f"[Audio Simulator] Starting stream (interval={interval}s)...")

    while True:
        result = simulate_audio_detection()
        count += 1
        result["stream_id"] = count
        print(f"[Audio #{count:04d}] 🔊 {result['class']} ({result['confidence']*100:.1f}%)")
        yield result

        if iterations is not None and count >= iterations:
            break

        time.sleep(interval)


if __name__ == "__main__":
    for result in run_audio_simulation(interval=2.0, iterations=5):
        pass
    print("\n[✓] Audio simulation complete.")
