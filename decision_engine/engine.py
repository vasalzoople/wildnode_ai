"""
wildnode_ai/decision_engine/engine.py
======================================
PURPOSE: The brain of WildNode AI. Combines:
  - Audio detection result
  - Vision detection result(s)
  - Environmental risk score
  → Into one final DECISION with an overall Risk Score and recommended action.

DECISION LOGIC (Rule-Based):
  IF high-priority animal detected + night + high env risk   → CRITICAL ALERT
  IF high-priority animal detected + high confidence          → HIGH ALERT
  IF medium-priority animal detected OR low confidence        → MEDIUM ALERT
  IF background only                                          → LOW (monitor)

BEGINNER NOTE:
  This is a "rule-based expert system" — a classic AI approach.
  Rules are written by domain experts (wildlife rangers, researchers).
  In future, this could be replaced by an LLM (GPT-4, Gemini) for
  more sophisticated contextual reasoning.
"""

import os
import sys
import json
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Risk Thresholds ─────────────────────────────────────────────────────────
THRESHOLD_CRITICAL = 75
THRESHOLD_HIGH     = 55
THRESHOLD_MEDIUM   = 30
# ─────────────────────────────────────────────────────────────────────────────

# ─── Animal Danger Weights (scale 0–1) ────────────────────────────────────────
ANIMAL_DANGER_WEIGHT = {
    "elephant"  : 1.00,   # Most dangerous due to size and crop damage
    "tiger"     : 0.95,   # Direct human threat
    "bear"      : 0.88,
    "wild boar" : 0.72,
    "leopard"   : 0.90,
    "zebra"     : 0.30,
    "giraffe"   : 0.20,
    "horse"     : 0.25,
    "wild dog"  : 0.60,
    "background": 0.00,
}

HIGH_PRIORITY_ANIMALS = {"elephant", "tiger", "bear", "leopard"}


def compute_risk_score(audio_result: dict,
                       vision_results: list,
                       env_risk: dict) -> dict:
    """
    Aggregate all sensor inputs into a single risk score.

    Score formula (weighted):
      audio_contribution  = confidence × animal_danger × 30
      vision_contribution = best_confidence × animal_danger × 40
      env_contribution    = env_risk_score × 0.30

    Total = capped at 100

    Args:
        audio_result   : From audio_detection/predict.py
        vision_results : List from vision_detection/detector.py
        env_risk       : From environmental/risk_calculator.py

    Returns:
        score_info : {
            "total_score"       : float (0–100),
            "risk_level"        : str,
            "audio_contribution": float,
            "vision_contribution": float,
            "env_contribution"  : float,
        }
    """

    # ── Audio Contribution ─────────────────────────────────────────────────
    audio_class = audio_result.get("class", "background")
    audio_conf  = audio_result.get("confidence", 0.0)
    audio_weight = ANIMAL_DANGER_WEIGHT.get(audio_class, 0.0)
    audio_score  = audio_conf * audio_weight * 30   # Max 30 points

    # ── Vision Contribution ────────────────────────────────────────────────
    vision_score = 0.0
    best_vision_class = "none"
    if vision_results:
        # Use the highest-confidence detection
        best = max(vision_results, key=lambda d: d["confidence"])
        best_vision_class = best["class"]
        vis_weight = ANIMAL_DANGER_WEIGHT.get(best["class"], 0.0)
        vision_score = best["confidence"] * vis_weight * 40   # Max 40 points

    # ── Environmental Contribution ─────────────────────────────────────────
    env_raw   = env_risk.get("score", 0.0)    # 0–100
    env_score = env_raw * 0.30               # Max 30 points

    # ── Combine ────────────────────────────────────────────────────────────
    total = min(100.0, audio_score + vision_score + env_score)

    # Determine level
    if total >= THRESHOLD_CRITICAL:
        level = "CRITICAL"
    elif total >= THRESHOLD_HIGH:
        level = "HIGH"
    elif total >= THRESHOLD_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "total_score"          : round(total, 1),
        "risk_level"           : level,
        "audio_contribution"   : round(audio_score, 1),
        "vision_contribution"  : round(vision_score, 1),
        "env_contribution"     : round(env_score, 1),
        "dominant_audio_class" : audio_class,
        "dominant_vision_class": best_vision_class,
    }


def make_decision(audio_result: dict,
                  vision_results: list,
                  env_risk: dict,
                  weather: dict,
                  location: str = "Sector 4") -> dict:
    """
    Core decision function – The main AI engine.

    Args:
        audio_result   : Audio detection prediction
        vision_results : Vision detection list
        env_risk       : Environmental risk dict
        weather        : Weather data dict
        location       : Sector/zone name for the alert message

    Returns:
        decision : Full decision dict with action, alert message, risk score
    """
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)

    # ── Compute aggregate risk ─────────────────────────────────────────────
    score_info = compute_risk_score(audio_result, vision_results, env_risk)
    risk_level = score_info["risk_level"]
    risk_score = score_info["total_score"]

    # ── Determine primary threat ───────────────────────────────────────────
    threats = []

    # Audio threat
    audio_class = audio_result.get("class", "background")
    audio_conf  = audio_result.get("confidence", 0.0)
    if audio_class != "background" and audio_conf > 0.50:
        threats.append({
            "source"    : "audio",
            "animal"    : audio_class,
            "confidence": audio_conf,
            "priority"  : "HIGH" if audio_class in HIGH_PRIORITY_ANIMALS else "MEDIUM"
        })

    # Vision threats
    for det in vision_results:
        if det["confidence"] > 0.45:
            threats.append({
                "source"    : "vision",
                "animal"    : det["class"],
                "confidence": det["confidence"],
                "priority"  : det.get("priority", "MEDIUM")
            })

    # ── Apply Decision Rules ───────────────────────────────────────────────
    primary_animal = None
    if threats:
        # Pick highest-priority, highest-confidence threat
        high_threats = [t for t in threats if t["priority"] == "HIGH"]
        if high_threats:
            primary_animal = max(high_threats, key=lambda t: t["confidence"])
        else:
            primary_animal = max(threats, key=lambda t: t["confidence"])

    # ── Generate Action & Alert Message ───────────────────────────────────
    emoji_map = {
        "elephant"  : "🐘",
        "tiger"     : "🐯",
        "bear"      : "🐻",
        "wild boar" : "🐗",
        "leopard"   : "🐆",
        "background": "🌿",
        "none"      : "🔍",
    }

    time_str = now.strftime("%I:%M %p")
    time_of_day = weather.get("time_of_day", "unknown")

    if risk_level == "CRITICAL":
        action = "IMMEDIATE EVACUATION – Alert all rangers + close roads"
        if primary_animal:
            animal = primary_animal["animal"]
            emoji  = emoji_map.get(animal, "🦁")
            conf   = primary_animal["confidence"] * 100
            alert_msg = (
                f"🚨 CRITICAL: {emoji} {animal.title()} detected in {location} "
                f"at {time_str} ({time_of_day}). Confidence: {conf:.1f}%. "
                f"Risk Score: {risk_score}/100. Immediate action required!"
            )
        else:
            alert_msg = f"🚨 CRITICAL environmental risk at {location}. Risk Score: {risk_score}/100."

    elif risk_level == "HIGH":
        action = "Alert rangers – Monitor closely + deploy patrol"
        if primary_animal:
            animal = primary_animal["animal"]
            emoji  = emoji_map.get(animal, "🦁")
            conf   = primary_animal["confidence"] * 100
            alert_msg = (
                f"⚠️ HIGH ALERT: {emoji} {animal.title()} detected near {location} "
                f"at {time_str}. Confidence: {conf:.1f}%. "
                f"Risk Score: {risk_score}/100. Stay indoors."
            )
        else:
            alert_msg = f"⚠️ HIGH ALERT: Unusual activity near {location}. Risk Score: {risk_score}/100."

    elif risk_level == "MEDIUM":
        action = "Log event – Increase monitoring frequency"
        if primary_animal:
            animal = primary_animal["animal"]
            emoji  = emoji_map.get(animal, "🦁")
            alert_msg = (
                f"📡 MEDIUM ALERT: {emoji} Possible {animal} activity near {location} "
                f"at {time_str}. Risk Score: {risk_score}/100. Monitoring..."
            )
        else:
            alert_msg = f"📡 MEDIUM: Background activity detected near {location}."

    else:  # LOW
        action = "Continue passive monitoring"
        alert_msg = f"✅ LOW RISK: Area clear at {location} ({time_str}). Score: {risk_score}/100."

    return {
        "timestamp"         : now.isoformat(),
        "location"          : location,
        "risk_level"        : risk_level,
        "risk_score"        : risk_score,
        "alert_message"     : alert_msg,
        "recommended_action": action,
        "threats_detected"  : threats,
        "score_breakdown"   : score_info,
        "weather_summary"   : {
            "condition"      : weather.get("condition", "unknown"),
            "time_of_day"    : time_of_day,
            "wind_speed_kmh" : weather.get("wind_speed_kmh", 0),
            "temperature_c"  : weather.get("temperature_c", 0),
        }
    }


if __name__ == "__main__":
    # Quick self-test
    from audio_detection.predict import simulate_audio_detection
    from vision_detection.detector import simulate_vision_detection
    from environmental.weather_api import get_mock_weather
    from environmental.risk_calculator import calculate_env_risk

    audio   = simulate_audio_detection()
    vision  = simulate_vision_detection()
    weather = get_mock_weather()
    env     = calculate_env_risk(weather)

    decision = make_decision(audio, vision, env, weather)

    print("\n" + "=" * 60)
    print("  WildNode AI – Decision Engine Test")
    print("=" * 60)
    print(f"  Risk Level    : {decision['risk_level']}")
    print(f"  Risk Score    : {decision['risk_score']} / 100")
    print(f"  Alert Message : {decision['alert_message']}")
    print(f"  Action        : {decision['recommended_action']}")
    print("=" * 60)
