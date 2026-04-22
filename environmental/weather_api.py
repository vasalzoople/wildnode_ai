"""
wildnode_ai/environmental/weather_api.py
==========================================
PURPOSE: Fetch real-time weather data from OpenWeather API and parse
         environmental risk factors for the decision engine.

HOW IT WORKS:
  1. If OPENWEATHER_API_KEY = "MOCK" → use offline mock data (default)
  2. Otherwise → call the real OpenWeather API and parse JSON response
  3. Extract: temperature, wind speed, wind direction, humidity, time of day
  4. Return structured weather data used by the risk calculator

API: https://openweathermap.org/api (Free tier: 1000 calls/day)
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Wind Direction Mapping ───────────────────────────────────────────────────
def degrees_to_cardinal(deg: float) -> str:
    """Convert wind degrees to compass direction (N, NE, E, SE, etc.)"""
    compass = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
               "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = round(deg / (360 / len(compass))) % len(compass)
    return compass[idx]


def get_time_of_day(hour: int) -> str:
    """Classify hour into time-of-day period."""
    if 5 <= hour < 8:   return "dawn"
    if 8 <= hour < 12:  return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 20: return "dusk"
    return "night"


# ─── Mock Weather Data ────────────────────────────────────────────────────────
def get_mock_weather(location: str = "Kaziranga") -> dict:
    """
    Generate realistic mock environmental data for offline use.
    Useful when no API key is available or for simulation mode.

    Returns data structure identical to what the real API returns.
    """
    import random

    # Current IST time
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = now.hour

    # Pick weather conditions (weighted toward realistic scenarios)
    conditions = ["Clear", "Clouds", "Rain", "Mist", "Fog"]
    condition  = random.choices(conditions, weights=[40, 25, 15, 10, 10])[0]

    wind_speed_kmh = round(random.uniform(2.5, 35.0), 1)
    wind_deg       = random.randint(0, 359)
    temperature    = round(random.uniform(16.0, 34.0), 1)
    humidity       = random.randint(45, 95)
    visibility_km  = round(random.uniform(0.5, 10.0), 1)

    return {
        "location"       : location,
        "timestamp"      : now.isoformat(),
        "hour"           : hour,
        "time_of_day"    : get_time_of_day(hour),
        "is_night"       : hour < 5 or hour >= 20,
        "is_dawn_dusk"   : (5 <= hour < 8) or (17 <= hour < 20),
        "temperature_c"  : temperature,
        "humidity_pct"   : humidity,
        "wind_speed_kmh" : wind_speed_kmh,
        "wind_direction" : degrees_to_cardinal(wind_deg),
        "wind_deg"       : wind_deg,
        "condition"      : condition,
        "visibility_km"  : visibility_km,
        "source"         : "mock",
    }


# ─── Real API ─────────────────────────────────────────────────────────────────
def get_real_weather(api_key: str, location: str = "Kaziranga") -> dict:
    """
    Fetch live weather data from OpenWeather API.

    Args:
        api_key  : Your free API key from openweathermap.org
        location : City name or "lat,lon" string

    Returns:
        Structured weather dict (same format as get_mock_weather)
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q"     : location,
        "appid" : api_key,
        "units" : "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse API response
        wind_deg       = data.get("wind", {}).get("deg", 0)
        wind_speed_ms  = data.get("wind", {}).get("speed", 0)
        wind_speed_kmh = round(wind_speed_ms * 3.6, 1)

        # UTC → IST
        ist = pytz.timezone("Asia/Kolkata")
        utc_dt = datetime.fromtimestamp(data["dt"], tz=timezone.utc)
        ist_dt = utc_dt.astimezone(ist)
        hour = ist_dt.hour

        return {
            "location"       : data["name"],
            "timestamp"      : ist_dt.isoformat(),
            "hour"           : hour,
            "time_of_day"    : get_time_of_day(hour),
            "is_night"       : hour < 5 or hour >= 20,
            "is_dawn_dusk"   : (5 <= hour < 8) or (17 <= hour < 20),
            "temperature_c"  : round(data["main"]["temp"], 1),
            "humidity_pct"   : data["main"]["humidity"],
            "wind_speed_kmh" : wind_speed_kmh,
            "wind_direction" : degrees_to_cardinal(wind_deg),
            "wind_deg"       : wind_deg,
            "condition"      : data["weather"][0]["main"],
            "visibility_km"  : round(data.get("visibility", 10000) / 1000, 1),
            "source"         : "openweather",
        }

    except requests.exceptions.Timeout:
        print("[WARN] Weather API timeout – falling back to mock")
        return get_mock_weather(location)
    except requests.exceptions.HTTPError as e:
        print(f"[WARN] Weather API error ({e}) – falling back to mock")
        return get_mock_weather(location)
    except Exception as e:
        print(f"[WARN] Weather fetch failed ({e}) – falling back to mock")
        return get_mock_weather(location)


# ─── Main Entry Point ─────────────────────────────────────────────────────────
def fetch_weather(location: str = None) -> dict:
    """
    Main function: automatically chooses real API or mock based on .env config.

    Returns:
        Structured weather dict
    """
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENWEATHER_API_KEY", "MOCK")
    loc     = location or os.getenv("WEATHER_LOCATION", "Kaziranga")

    if api_key.upper() == "MOCK" or not api_key:
        return get_mock_weather(loc)
    else:
        return get_real_weather(api_key, loc)


if __name__ == "__main__":
    weather = fetch_weather()
    print(json.dumps(weather, indent=2))
