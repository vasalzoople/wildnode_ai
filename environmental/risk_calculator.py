"""
wildnode_ai/environmental/risk_calculator.py
=============================================
PURPOSE: Calculate an environmental risk score (0–100) based on weather
         conditions that affect wildlife activity and detection difficulty.

RISK FACTORS:
  - Time of night    → Animals more active (high risk)
  - Low visibility   → Hard to see / flee (higher danger)
  - Strong wind      → Carries scent, affects sound detection
  - Dawn/Dusk        → Twilight = peak wildlife movement zone
  - Rainy/Foggy      → Harder to detect, animals seek shelter near edges
  - High humidity    → Common in forest areas during monsoon (activity spike)
"""

def calculate_env_risk(weather: dict) -> dict:
    """
    Compute an environmental risk score from weather data.

    Args:
        weather : Dict from weather_api.fetch_weather()

    Returns:
        env_risk : {
            "score"   : float (0–100),
            "level"   : "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "factors" : list of str (reasons for score),
        }
    """
    score   = 0.0
    factors = []

    # ── Factor 1: Time of Night ────────────────────────────────────────────
    if weather.get("is_night", False):
        score += 35
        factors.append("🌙 Night time (peak wildlife activity)")
    elif weather.get("is_dawn_dusk", False):
        score += 20
        factors.append("🌅 Dawn/Dusk (twilight movement zone)")
    else:
        score += 5
        factors.append("☀️ Daytime (lower wildlife activity)")

    # ── Factor 2: Visibility ──────────────────────────────────────────────
    visibility = weather.get("visibility_km", 10.0)
    if visibility < 1.0:
        score += 25
        factors.append(f"🌫️ Very low visibility ({visibility} km) – high danger")
    elif visibility < 3.0:
        score += 15
        factors.append(f"🌁 Reduced visibility ({visibility} km)")
    elif visibility < 6.0:
        score += 7
        factors.append(f"🔭 Moderate visibility ({visibility} km)")

    # ── Factor 3: Wind Speed ──────────────────────────────────────────────
    wind = weather.get("wind_speed_kmh", 0)
    if wind > 30:
        score += 15
        factors.append(f"💨 Strong winds ({wind} km/h) – scent dispersal")
    elif wind > 15:
        score += 8
        factors.append(f"🌬️ Moderate winds ({wind} km/h)")
    else:
        score += 2
        factors.append(f"🍃 Calm winds ({wind} km/h)")

    # ── Factor 4: Weather Condition ───────────────────────────────────────
    condition = weather.get("condition", "Clear").lower()
    if "rain" in condition or "drizzle" in condition:
        score += 12
        factors.append("🌧️ Rainfall – animals move toward forest edges")
    elif "fog" in condition or "mist" in condition:
        score += 18
        factors.append("🌫️ Foggy – very low visibility, high encroachment risk")
    elif "thunder" in condition or "storm" in condition:
        score += 20
        factors.append("⛈️ Storm – erratic animal movements")
    elif "cloud" in condition:
        score += 5
        factors.append("☁️ Clouds – neutral condition")

    # ── Factor 5: Humidity ────────────────────────────────────────────────
    humidity = weather.get("humidity_pct", 50)
    if humidity > 85:
        score += 8
        factors.append(f"💧 High humidity ({humidity}%) – monsoon season patterns")
    elif humidity > 65:
        score += 3

    # Clamp to [0, 100]
    score = min(100.0, max(0.0, score))

    # Determine risk level
    if score >= 75:
        level = "CRITICAL"
    elif score >= 55:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "score"   : round(score, 1),
        "level"   : level,
        "factors" : factors,
    }


if __name__ == "__main__":
    from weather_api import get_mock_weather
    import json

    weather = get_mock_weather()
    result  = calculate_env_risk(weather)

    print(f"\nWeather: {weather['condition']} | {weather['time_of_day']} | {weather['wind_speed_kmh']} km/h wind")
    print(f"Env Risk Score : {result['score']} / 100  →  {result['level']}")
    print("Factors:")
    for f in result["factors"]:
        print(f"  {f}")
