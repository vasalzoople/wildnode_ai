"""
wildnode_ai/pipeline/main_pipeline.py
=======================================
PURPOSE: Command-line entry point that runs the FULL WildNode AI pipeline
         in a continuous loop:

  SIMULATE → DETECT (Audio+Vision) → WEATHER → DECIDE → ALERT → LOG → REPEAT

HOW TO RUN:
  python pipeline/main_pipeline.py                      # Run indefinitely
  python pipeline/main_pipeline.py --cycles 10          # Run 10 cycles
  python pipeline/main_pipeline.py --interval 5         # 5 second between cycles
"""

import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from audio_detection.predict import simulate_audio_detection
from vision_detection.detector import simulate_vision_detection
from environmental.weather_api import fetch_weather
from environmental.risk_calculator import calculate_env_risk
from decision_engine.engine import make_decision
from alert_system.alerter import dispatch_alert


BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🌿  WildNode AI – Wildlife Monitor v1.0          ║
║     Edge AI-Based Human-Wildlife Conflict Detection      ║
╚══════════════════════════════════════════════════════════╝
"""


def run_pipeline(cycles: int = None, interval: float = 6.0,
                 sector: str = "Sector 4 – Kaziranga"):
    """
    Run the full WildNode AI detection pipeline.

    Args:
        cycles   : How many cycles to run (None = infinite)
        interval : Seconds between cycles
        sector   : Monitoring location name
    """
    print(BANNER)
    print(f"  📍 Sector  : {sector}")
    print(f"  ⏱ Interval: {interval}s between cycles")
    print(f"  🔁 Cycles  : {'∞ (Ctrl+C to stop)' if cycles is None else cycles}")
    print(f"\n{'─'*60}\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            if cycles is not None and cycle > cycles:
                break

            print(f"\n[Cycle {cycle:04d}] {'─'*45}")

            # Step 1: Audio detection
            audio = simulate_audio_detection()
            print(f"  🔊 Audio   : {audio['class']} ({audio['confidence']*100:.1f}%)")

            # Step 2: Vision detection
            vision = simulate_vision_detection()
            if vision:
                for v in vision:
                    print(f"  📷 Vision  : {v['label']}")
            else:
                print(f"  📷 Vision  : Clear frame")

            # Step 3: Environmental data
            weather = fetch_weather()
            print(f"  🌦 Weather : {weather['condition']} | {weather['time_of_day']} | "
                  f"{weather['wind_speed_kmh']}km/h wind")

            # Step 4: Environmental risk
            env = calculate_env_risk(weather)
            print(f"  🌡 Env Risk: {env['level']} ({env['score']}/100)")

            # Step 5: Decision engine
            decision = make_decision(audio, vision, env, weather, location=sector)
            level = decision["risk_level"]
            score = decision["risk_score"]

            # Step 6: Alert dispatch (console + log)
            dispatch_alert(decision, enable_whatsapp=(level in ("HIGH","CRITICAL")),
                           enable_sms=(level == "CRITICAL"))

            # Wait before next cycle
            if cycles is None or cycle < cycles:
                print(f"  ⏳ Next cycle in {interval}s...")
                time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n[INFO] Pipeline stopped by user after {cycle} cycles.")
        print("[INFO] Check alert_system/alert_log.json for full logs.")
        print("[INFO] Run dashboard with: streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildNode AI – Main Pipeline")
    parser.add_argument("--cycles",   type=int,   default=None, help="Number of cycles (default: infinite)")
    parser.add_argument("--interval", type=float, default=6.0,  help="Seconds between cycles (default: 6)")
    parser.add_argument("--sector",   type=str,   default="Sector 4 – Kaziranga", help="Sector name")
    args = parser.parse_args()

    run_pipeline(cycles=args.cycles, interval=args.interval, sector=args.sector)
