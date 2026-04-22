"""
wildnode_ai/alert_system/alerter.py
=====================================
PURPOSE: Dispatch alerts via multiple simulated channels.

CHANNELS:
  1. Console   → Colored terminal output with risk emoji
  2. Mock SMS  → Simulated Twilio SMS (printed to console)
  3. Mock WA   → Simulated WhatsApp message (formatted print)
  4. JSON Log  → Persistent log file (alert_log.json)

BEGINNER NOTE:
  In a real production system, you would:
  - Use Twilio API for real SMS/WhatsApp  (twilio.com)
  - Use Firebase Cloud Messaging for push notifications
  - Use email via SMTP (smtplib / SendGrid)
  This module simulates all of these without needing real credentials.
"""

import os
import sys
import json
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ALERT_LOG_PATH = os.path.join(BASE_DIR, "alert_log.json")

# ANSI escape codes for colored terminal output
COLORS = {
    "CRITICAL" : "\033[91m",   # Bright Red
    "HIGH"     : "\033[38;5;208m",  # Orange
    "MEDIUM"   : "\033[93m",   # Yellow
    "LOW"      : "\033[92m",   # Green
    "RESET"    : "\033[0m",
    "BOLD"     : "\033[1m",
    "CYAN"     : "\033[96m",
    "DIM"      : "\033[2m",
}

RISK_ICONS = {
    "CRITICAL" : "🚨",
    "HIGH"     : "⚠️ ",
    "MEDIUM"   : "📡",
    "LOW"      : "✅",
}


def _load_alert_log() -> list:
    """Load existing alert log from JSON file."""
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_alert_log(alerts: list):
    """Save alert log to JSON file. Keep last 200 records."""
    try:
        with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(alerts[-200:], f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Could not save alert log: {e}")


def send_console_alert(decision: dict):
    """
    Print a formatted, colored alert to the terminal.

    Args:
        decision : Decision dict from decision_engine/engine.py
    """
    level    = decision.get("risk_level", "LOW")
    score    = decision.get("risk_score", 0)
    msg      = decision.get("alert_message", "")
    action   = decision.get("recommended_action", "")
    location = decision.get("location", "Unknown")
    ts       = decision.get("timestamp", "")

    color  = COLORS.get(level, COLORS["RESET"])
    icon   = RISK_ICONS.get(level, "🔍")
    reset  = COLORS["RESET"]
    bold   = COLORS["BOLD"]
    dim    = COLORS["DIM"]
    cyan   = COLORS["CYAN"]

    border = "─" * 60
    print(f"\n{color}{bold}{border}{reset}")
    print(f"{color}{bold}  {icon} WildNode AI ALERT  [{level}]  Score: {score}/100{reset}")
    print(f"{color}{border}{reset}")
    print(f"  {bold}Message :{reset} {msg}")
    print(f"  {bold}Action  :{reset} {cyan}{action}{reset}")
    print(f"  {bold}Location:{reset} {location}")
    print(f"  {dim}Time    : {ts}{reset}")
    print(f"{color}{border}{reset}\n")


def send_mock_whatsapp(decision: dict, phone: str = "+91XXXXXXXXXX"):
    """
    Simulate a WhatsApp message alert (printed to console).
    Real implementation would use WhatsApp Cloud API or Twilio.

    Args:
        decision : Decision dict
        phone    : Recipient phone number (mock)
    """
    msg   = decision.get("alert_message", "")
    level = decision.get("risk_level", "LOW")
    score = decision.get("risk_score", 0)

    print(f"\n{COLORS['CYAN']}{COLORS['BOLD']}  📱 Simulated WhatsApp Alert{COLORS['RESET']}")
    print(f"  {'─' * 40}")
    print(f"  To     : {phone}")
    print(f"  Via    : WhatsApp Cloud API (simulated)")
    print(f"  Status : ✅ Message Sent")
    print(f"\n  ╔══════════════════════════════════╗")
    print(f"  ║  🌿 WildNode AI Alert System     ║")
    print(f"  ╠══════════════════════════════════╣")
    print(f"  ║  {msg[:32]}")
    if len(msg) > 32:
        print(f"  ║  {msg[32:64]}")
    if len(msg) > 64:
        print(f"  ║  {msg[64:96]}")
    print(f"  ║  Risk Level: {level:<20}║")
    print(f"  ║  Score     : {score}/100{' ' * 14}║")
    print(f"  ║  Reply STOP to unsubscribe       ║")
    print(f"  ╚══════════════════════════════════╝")


def send_mock_sms(decision: dict, phone: str = "+91XXXXXXXXXX"):
    """
    Simulate an SMS alert (Twilio-style mock).

    Args:
        decision : Decision dict
        phone    : Recipient phone number
    """
    msg   = decision.get("alert_message", "")
    level = decision.get("risk_level", "LOW")

    # Truncate SMS to 160 chars (real SMS limit)
    sms_body = f"WildNodeAI | {level}: {msg}"[:160]

    print(f"\n  📟 Simulated SMS (Twilio mock)")
    print(f"  To      : {phone}")
    print(f"  Message : {sms_body}")
    print(f"  Status  : Delivered ✅  [SID: WN{hash(msg) % 999999:06d}]")


def log_alert(decision: dict):
    """
    Append the decision/alert to the persistent JSON log file.

    Args:
        decision : Decision dict
    """
    alerts = _load_alert_log()
    log_entry = {
        "id"               : len(alerts) + 1,
        "timestamp"        : decision.get("timestamp"),
        "location"         : decision.get("location"),
        "risk_level"       : decision.get("risk_level"),
        "risk_score"       : decision.get("risk_score"),
        "alert_message"    : decision.get("alert_message"),
        "recommended_action": decision.get("recommended_action"),
        "threats_detected" : decision.get("threats_detected", []),
        "weather_summary"  : decision.get("weather_summary", {}),
    }
    alerts.append(log_entry)
    _save_alert_log(alerts)
    return log_entry


def dispatch_alert(decision: dict, phone: str = "+91XXXXXXXXXX",
                   enable_whatsapp: bool = True,
                   enable_sms: bool = True,
                   enable_log: bool = True) -> dict:
    """
    Master alert dispatcher – calls all alert channels.

    Args:
        decision        : Decision dict from engine.py
        phone           : Phone number (mock)
        enable_whatsapp : Send simulated WhatsApp message
        enable_sms      : Send simulated SMS
        enable_log      : Write to JSON log file

    Returns:
        log_entry : The saved log entry dict
    """
    risk_level = decision.get("risk_level", "LOW")

    # Always send console alert
    send_console_alert(decision)

    # Only send WhatsApp/SMS for MEDIUM and above
    if risk_level in ("MEDIUM", "HIGH", "CRITICAL"):
        if enable_whatsapp:
            send_mock_whatsapp(decision, phone)
        if enable_sms:
            send_mock_sms(decision, phone)

    # Always log
    log_entry = None
    if enable_log:
        log_entry = log_alert(decision)

    return log_entry


def get_recent_alerts(n: int = 20) -> list:
    """Retrieve the last N alerts from the log file."""
    alerts = _load_alert_log()
    return alerts[-n:]


if __name__ == "__main__":
    # Self-test: generate a mock decision and dispatch alerts
    from decision_engine.engine import make_decision
    from audio_detection.predict import simulate_audio_detection
    from vision_detection.detector import simulate_vision_detection
    from environmental.weather_api import get_mock_weather
    from environmental.risk_calculator import calculate_env_risk

    audio   = simulate_audio_detection()
    vision  = simulate_vision_detection()
    weather = get_mock_weather()
    env     = calculate_env_risk(weather)
    decision = make_decision(audio, vision, env, weather)

    print("\n🔁 Running Alert System Test...\n")
    dispatch_alert(decision)
    print(f"\n[✓] Alert logged to: {ALERT_LOG_PATH}")
