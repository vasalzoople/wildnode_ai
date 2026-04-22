import sys
sys.path.insert(0, '.')
from audio_detection.predict import simulate_audio_detection
from vision_detection.detector import simulate_vision_detection
from environmental.weather_api import fetch_weather
from environmental.risk_calculator import calculate_env_risk
from decision_engine.engine import make_decision
from alert_system.alerter import dispatch_alert

print('[OK] All imports succeeded!')
a = simulate_audio_detection()
v = simulate_vision_detection()
w = fetch_weather()
e = calculate_env_risk(w)
d = make_decision(a, v, e, w)
print(f"[OK] Risk Level : {d['risk_level']} | Score = {d['risk_score']}")
print(f"     Audio      : {a['class']} ({a['confidence']*100:.1f}%)")
print(f"     Vision     : {[x['class'] for x in v] if v else 'clear'}")
print(f"     Alert      : {d['alert_message'][:70]}")
