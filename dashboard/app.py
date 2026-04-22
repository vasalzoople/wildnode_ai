"""
wildnode_ai/dashboard/app.py
==============================
WildNode AI – Glassmorphism Streamlit Dashboard

HOW TO RUN:
  streamlit run dashboard/app.py

FEATURES:
  - Live simulation feed (auto-refreshing)
  - Risk score gauge (Plotly)
  - Alert log table
  - Environmental weather widget
  - Detection confidence bar charts
  - Glassmorphism dark theme with animated elements
  - 6 tabs: Overview | Audio | Vision | Alerts | Logs | About
"""

import os
import sys
import json
import random
import time
from datetime import datetime
import pytz

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import queue
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, AudioProcessorBase

# Database for detailed info
ANIMAL_INFO_DB = {
    'elephant': {'status': 'Endangered', 'behavior': 'Highly intelligent, travels in herds. Can be aggressive when protecting young.', 'action': 'Deploy heavy unit. Keep extreme distance.', 'diet': 'Herbivore', 'emoji':'🐘'},
    'tiger': {'status': 'Endangered', 'behavior': 'Solitary apex predator, highly territorial.', 'action': 'Immediate evacuation. Patrol requested.', 'diet': 'Carnivore', 'emoji':'🐯'},
    'bear': {'status': 'Vulnerable', 'behavior': 'Unpredictable, highly protective of cubs.', 'action': 'Sound loud alarms. Do not approach.', 'diet': 'Omnivore', 'emoji':'🐻'},
    'wild boar': {'status': 'Least Concern', 'behavior': 'Aggressive when cornered.', 'action': 'Disperse with sound deterrents. Secure crops.', 'diet': 'Omnivore', 'emoji':'🐗'},
    'zebra': {'status': 'Near Threatened', 'behavior': 'Flighty, travels in herds.', 'action': 'Monitor and log. Ensure safe passage.', 'diet': 'Herbivore', 'emoji':'🦓'},
    'giraffe': {'status': 'Vulnerable', 'behavior': 'Peaceful but powerful kicks.', 'action': 'Monitor and log. Prevent traffic collision.', 'diet': 'Herbivore', 'emoji':'🦒'},
    'leopard': {'status': 'Vulnerable', 'behavior': 'Stealthy, nocturnal hunter.', 'action': 'Secure livestock. Alert villagers.', 'diet': 'Carnivore', 'emoji':'🐆'},
    'wild dog': {'status': 'Endangered', 'behavior': 'Pack hunters, highly coordinated.', 'action': 'Secure livestock. Ranger patrol requested.', 'diet': 'Carnivore', 'emoji':'🐺'},
    'horse': {'status': 'Domestic', 'behavior': 'Flocking domestic species.', 'action': 'Ensure safe environment boundaries.', 'diet': 'Herbivore', 'emoji':'🐴'},
    'cow': {'status': 'Domestic', 'behavior': 'Grazing domestic species.', 'action': 'Log tracking stats.', 'diet': 'Herbivore', 'emoji':'🐮'},
    'cat': {'status': 'Domestic', 'behavior': 'Small free-ranging feline.', 'action': 'None.', 'diet': 'Carnivore', 'emoji':'🐱'},
    'dog': {'status': 'Domestic', 'behavior': 'Domesticated canine.', 'action': 'Return to owner.', 'diet': 'Omnivore', 'emoji':'🐶'}
}

if 'GLOBAL_VISION_RES' not in globals():
    global GLOBAL_VISION_RES
    GLOBAL_VISION_RES = []
if 'GLOBAL_AUDIO_RES' not in globals():
    global GLOBAL_AUDIO_RES
    GLOBAL_AUDIO_RES = None

from environmental.weather_api import fetch_weather
from environmental.risk_calculator import calculate_env_risk
from decision_engine.engine import make_decision
from alert_system.alerter import dispatch_alert, get_recent_alerts, ALERT_LOG_PATH

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WildNode AI – Wildlife Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

IST = pytz.timezone("Asia/Kolkata")

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  –  Glassmorphism Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: radial-gradient(ellipse at 20% 50%, #0d2137 0%, #050d1a 40%, #000a14 100%);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(6,18,34,0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(56,189,248,0.15);
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom: 16px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 20px rgba(56,189,248,0.08);
    }

    /* ── Risk Badge ── */
    .badge-critical { background: linear-gradient(135deg,#7f1d1d,#dc2626); color:#fff; }
    .badge-high     { background: linear-gradient(135deg,#7c2d12,#ea580c); color:#fff; }
    .badge-medium   { background: linear-gradient(135deg,#713f12,#ca8a04); color:#fff; }
    .badge-low      { background: linear-gradient(135deg,#14532d,#16a34a); color:#fff; }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px; border-radius: 20px;
        font-size: 12px; font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    /* ── Pulse animation for CRITICAL ── */
    @keyframes pulse-ring {
        0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.6); }
        70%  { box-shadow: 0 0 0 16px rgba(220,38,38,0); }
        100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
    }
    .pulse { animation: pulse-ring 1.4s infinite; }

    /* ── Floating particles ── */
    @keyframes float-up {
        0%   { transform: translateY(0) scale(1); opacity: 0.6; }
        100% { transform: translateY(-60px) scale(0); opacity: 0; }
    }

    /* ── Stat Metric Cards ── */
    .metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1; }
    .metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.55; margin-top: 4px; }
    .metric-delta { font-size: 0.8rem; margin-top: 6px; }

    /* ── Timeline item ── */
    .timeline-item {
        border-left: 2px solid rgba(56,189,248,0.35);
        padding: 8px 0 8px 16px;
        margin-bottom: 8px;
        position: relative;
    }
    .timeline-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #38bdf8;
        position: absolute; left: -5px; top: 12px;
    }

    /* ── Streamlit override tweaks ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px; padding: 4px;
        border: 1px solid rgba(255,255,255,0.08);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; color: rgba(255,255,255,0.55);
        font-weight: 500; padding: 8px 18px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.15) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56,189,248,0.3) !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: rgba(255,255,255,0.5) !important; font-size: 0.7rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(99,102,241,0.15));
        border: 1px solid rgba(56,189,248,0.35);
        color: #38bdf8; border-radius: 10px; font-weight: 600;
        transition: all 0.25s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(56,189,248,0.3), rgba(99,102,241,0.3));
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(56,189,248,0.25);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #050d1a; }
    ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 3px; }

    h1,h2,h3 { color: #e2e8f0 !important; }
    p, li { color: rgba(226,232,240,0.75) !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def risk_color(level: str) -> str:
    return {"CRITICAL": "#dc2626", "HIGH": "#ea580c", "MEDIUM": "#ca8a04", "LOW": "#16a34a"}.get(level, "#64748b")

def risk_emoji(level: str) -> str:
    return {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "📡", "LOW": "✅"}.get(level, "🔍")

def animal_emoji(name: str) -> str:
    return {"elephant": "🐘", "tiger": "🐯", "bear": "🐻", "wild boar": "🐗",
            "zebra": "🦓", "giraffe": "🦒", "background": "🌿"}.get(name, "🦁")


def make_gauge(score: float, level: str) -> go.Figure:
    color = risk_color(level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 26, "color": "#e2e8f0", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(22,163,74,0.12)"},
                {"range": [30, 55], "color": "rgba(202,138,4,0.12)"},
                {"range": [55, 75], "color": "rgba(234,88,12,0.12)"},
                {"range": [75, 100],"color": "rgba(220,38,38,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#94a3b8"},
    )
    return fig


def make_confidence_bar(audio_result: dict, vision_results: list) -> go.Figure:
    labels, values, colors = [], [], []
    for cls, score in audio_result.get("all_scores", {}).items():
        labels.append(f"🔊 {cls}")
        values.append(round(score * 100, 1))
        colors.append("#38bdf8")
    for det in vision_results[:3]:
        labels.append(f"📷 {det['class']}")
        values.append(round(det['confidence'] * 100, 1))
        colors.append("#a78bfa")

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.08)", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside", textfont=dict(color="#cbd5e1", size=11),
    ))
    fig.update_layout(
        height=max(160, len(labels) * 35 + 60),
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(range=[0, 110], showgrid=False, zeroline=False,
                   tickfont=dict(color="#64748b"), showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#cbd5e1", size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_history_chart(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    df = pd.DataFrame(history[-40:])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["risk_score"],
        fill="tozeroy", fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="#38bdf8", width=2),
        mode="lines+markers",
        marker=dict(size=5, color=[risk_color(l) for l in df["risk_level"]]),
        hovertemplate="Score: %{y}<br>%{text}<extra></extra>",
        text=df["risk_level"].tolist(),
    ))
    fig.update_layout(
        height=160, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 105], showgrid=True,
                   gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#64748b", size=9)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running"      : True,
        "history"      : [],
        "last_decision": None,
        "last_audio"   : None,
        "last_vision"  : [],
        "last_weather" : None,
        "last_env"     : None,
        "total_detections": 0,
        "high_alerts"  : 0,
        "sector"       : "Sector 4 – Kaziranga",
        "cycle"        : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
#  RUN ONE DETECTION CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def process_live_data(audio_res, vision_res):
    audio = audio_res if audio_res else {"class": "background", "confidence": 1.0, "all_scores": {}}
    vision = vision_res if vision_res else []
    weather = fetch_weather()
    env     = calculate_env_risk(weather)
    decision = make_decision(audio, vision, env, weather,
                              location=st.session_state.sector)

    # Dispatch (log only, no console spam in dashboard)
    dispatch_alert(decision, enable_whatsapp=False, enable_sms=False, enable_log=True)

    st.session_state.last_audio    = audio
    st.session_state.last_vision   = vision
    st.session_state.last_weather  = weather
    st.session_state.last_env      = env
    st.session_state.last_decision = decision
    st.session_state.cycle        += 1
    st.session_state.total_detections += 1 + len(vision)
    if decision["risk_level"] in ("HIGH", "CRITICAL"):
        st.session_state.high_alerts += 1

    st.session_state.history.append({
        "risk_score" : decision["risk_score"],
        "risk_level" : decision["risk_level"],
        "timestamp"  : decision["timestamp"],
    })
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px'>
          <div style='font-size:2.5rem'>🌿</div>
          <div style='font-size:1.3rem; font-weight:700; color:#38bdf8; letter-spacing:1px'>WildNode AI</div>
          <div style='font-size:0.7rem; color:#64748b; letter-spacing:2px; text-transform:uppercase'>Edge Wildlife Monitor</div>
        </div>
        <hr style='border-color:rgba(255,255,255,0.06); margin:0 0 20px'>
        """, unsafe_allow_html=True)

        # Control buttons


        # Settings
        st.markdown("<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>⚙ Settings</div>", unsafe_allow_html=True)
        st.session_state.sector = st.selectbox(
            "Monitoring Sector",
            ["Sector 4 – Kaziranga", "Sector 7 – Jim Corbett", "Zone B – Bandipur",
             "Block 3 – Ranthambore", "Unit 9 – Sundarbans"],
            label_visibility="collapsed"
        )

        refresh_rate = st.slider("UI Refresh Rate (s)", 1, 10, 2)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:14px 0'>", unsafe_allow_html=True)

        # Live status indicator
        status_color = "#16a34a"
        status_label = "LIVE"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px'>
          <div style='width:8px;height:8px;border-radius:50%;background:{status_color};
                      box-shadow:0 0 6px {status_color};'></div>
          <span style='font-size:0.75rem;color:{status_color};font-weight:600;letter-spacing:1px'>{status_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        st.metric("Total Cycles", st.session_state.cycle)
        st.metric("Total Detections", st.session_state.total_detections)
        st.metric("High/Critical Alerts", st.session_state.high_alerts)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:14px 0'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.65rem; color:#334155; text-align:center; line-height:1.6'>
          📡 Simulated Edge Node v1.0<br>
          🛰 Data: OpenWeather + YOLOv8n<br>
          🧠 Model: CNN + Rule Engine
        </div>
        """, unsafe_allow_html=True)

    return refresh_rate


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview():
    d = st.session_state.last_decision
    audio   = st.session_state.last_audio
    vision  = st.session_state.last_vision
    weather = st.session_state.last_weather

    if not d:
        st.markdown("""
        <div class='glass-card' style='text-align:center; padding:60px 20px'>
          <div style='font-size:3rem; margin-bottom:12px'>🌿</div>
          <div style='font-size:1.2rem; color:#38bdf8; font-weight:600'>System Ready</div>
          <div style='color:#64748b; margin-top:8px'>Click <b>▶ Start</b> or <b>🔄 Run Once</b> to begin monitoring</div>
        </div>
        """, unsafe_allow_html=True)
        return

    level = d["risk_level"]
    score = d["risk_score"]
    color = risk_color(level)
    emoji = risk_emoji(level)
    pulse_cls = "pulse" if level == "CRITICAL" else ""

    # ── Alert banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='glass-card {pulse_cls}'
         style='border-color:{color}44; border-left:4px solid {color};'>
      <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap'>
        <span style='font-size:1.8rem'>{emoji}</span>
        <div style='flex:1'>
          <div style='font-size:0.65rem;color:{color};text-transform:uppercase;
                      letter-spacing:1.5px;font-weight:700;margin-bottom:3px'>
            {level} ALERT
          </div>
          <div style='color:#e2e8f0; font-size:0.95rem; font-weight:500'>
            {d['alert_message']}
          </div>
        </div>
        <div style='background:{color}22; border:1px solid {color}55;
                    border-radius:10px; padding:8px 16px; text-align:center'>
          <div style='font-size:1.6rem; font-weight:700; color:{color}'>{score}</div>
          <div style='font-size:0.6rem; color:{color}99; text-transform:uppercase'>Risk Score</div>
        </div>
      </div>
      <div style='margin-top:10px; font-size:0.8rem; color:#64748b'>
        🎯 Action: <span style='color:#94a3b8'>{d['recommended_action']}</span> &nbsp;|&nbsp;
        📍 {d['location']} &nbsp;|&nbsp;
        🕐 {d['timestamp'][11:19]} IST
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metrics row ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🔊 Audio Detection",
                  f"{animal_emoji(audio['class'])} {audio['class'].title()}",
                  f"{audio['confidence']*100:.0f}% confidence")
    with c2:
        v_label = f"{animal_emoji(vision[0]['class'])} {vision[0]['class'].title()}" if vision else "None"
        v_delta = f"{vision[0]['confidence']*100:.0f}% conf" if vision else "Clear frame"
        st.metric("📷 Vision Detection", v_label, v_delta)
    with c3:
        env = st.session_state.last_env
        st.metric("🌡 Env Risk", f"{env['level']}", f"{env['score']:.0f}/100")
    with c4:
        st.metric("💨 Wind Speed",
                  f"{weather.get('wind_speed_kmh',0):.0f} km/h",
                  weather.get("wind_direction", "N/A"))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Gauge + History ───────────────────────────────────────────────────
    col_gauge, col_hist = st.columns([1, 2])
    with col_gauge:
        st.markdown("<div class='glass-card' style='padding:16px'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px'>Risk Gauge</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(score, level), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div style='text-align:center;margin-top:-10px'><span class='risk-badge badge-{level.lower()}'>{level}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_hist:
        st.markdown("<div class='glass-card' style='padding:16px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Risk Score History</div>", unsafe_allow_html=True)
        if st.session_state.history:
            st.plotly_chart(make_history_chart(st.session_state.history),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("<div style='color:#334155;text-align:center;padding:40px'>No history yet</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Confidence bars + Weather ─────────────────────────────────────────
    col_conf, col_wx = st.columns([3, 2])
    with col_conf:
        st.markdown("<div class='glass-card' style='padding:16px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Detection Confidence</div>", unsafe_allow_html=True)
        st.plotly_chart(make_confidence_bar(audio, vision),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_wx:
        st.markdown(f"""
        <div class='glass-card' style='padding:18px'>
          <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px'>🌦 Environmental</div>
          <div style='display:grid; grid-template-columns:1fr 1fr; gap:10px'>
            <div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px'>
              <div style='font-size:1.3rem;font-weight:700;color:#38bdf8'>{weather.get('temperature_c','?')}°C</div>
              <div style='font-size:0.65rem;color:#64748b'>Temperature</div>
            </div>
            <div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px'>
              <div style='font-size:1.3rem;font-weight:700;color:#a78bfa'>{weather.get('humidity_pct','?')}%</div>
              <div style='font-size:0.65rem;color:#64748b'>Humidity</div>
            </div>
            <div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px'>
              <div style='font-size:1.3rem;font-weight:700;color:#34d399'>{weather.get('wind_speed_kmh','?')} km/h</div>
              <div style='font-size:0.65rem;color:#64748b'>Wind Speed</div>
            </div>
            <div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:10px'>
              <div style='font-size:1.3rem;font-weight:700;color:#fbbf24'>{weather.get('visibility_km','?')} km</div>
              <div style='font-size:0.65rem;color:#64748b'>Visibility</div>
            </div>
          </div>
          <div style='margin-top:10px; font-size:0.8rem; color:#94a3b8'>
            ⛅ {weather.get('condition','?')} &nbsp;·&nbsp; {weather.get('time_of_day','?').title()} &nbsp;·&nbsp;
            {'🌙 Night' if weather.get('is_night') else '☀️ Day'}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Env factors ───────────────────────────────────────────────────────
    env = st.session_state.last_env
    if env and env.get("factors"):
        factor_html = "".join(
            f"<span style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);"
            f"border-radius:6px;padding:4px 10px;font-size:0.78rem;color:#94a3b8'>{f}</span>"
            for f in env["factors"]
        )
        st.markdown(f"""
        <div class='glass-card' style='padding:14px 18px'>
          <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>
            🔍 Environmental Risk Factors
          </div>
          <div style='display:flex;flex-wrap:wrap;gap:8px'>{factor_html}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: ALERTS LOG
# ─────────────────────────────────────────────────────────────────────────────
def tab_alerts():
    alerts = get_recent_alerts(50)
    if not alerts:
        st.markdown("<div class='glass-card' style='text-align:center;padding:40px;color:#64748b'>No alerts logged yet.</div>", unsafe_allow_html=True)
        return

    alerts_rev = list(reversed(alerts))
    for a in alerts_rev[:20]:
        lvl = a.get("risk_level", "LOW")
        col = risk_color(lvl)
        ico = risk_emoji(lvl)
        ts  = a.get("timestamp", "")[:19].replace("T", " ")
        st.markdown(f"""
        <div class='glass-card' style='padding:14px 18px;border-left:3px solid {col};margin-bottom:8px'>
          <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap'>
            <span style='font-size:1.1rem'>{ico}</span>
            <span class='risk-badge badge-{lvl.lower()}'>{lvl}</span>
            <span style='color:#64748b;font-size:0.75rem'>{ts}</span>
            <span style='color:#64748b;font-size:0.75rem'>📍 {a.get('location','?')}</span>
            <span style='margin-left:auto;color:{col};font-weight:700'>{a.get('risk_score',0)}/100</span>
          </div>
          <div style='color:#cbd5e1;font-size:0.85rem;margin-top:6px'>{a.get('alert_message','')}</div>
          <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>🎯 {a.get('recommended_action','')}</div>
        </div>
        """, unsafe_allow_html=True)

    if len(alerts) > 20:
        st.info(f"Showing 20 most recent of {len(alerts)} total alerts.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB: AUDIO DETAIL (upgraded: simulation + real audio upload)
# ─────────────────────────────────────────────────────────────────────────────
def tab_audio():
    st.markdown("""
    <div class='glass-card' style='padding:14px 18px;margin-bottom:4px'>
      <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
        🔊 Audio Detection System
      </div>
      <div style='color:#94a3b8;font-size:0.82rem;line-height:1.7'>
        <b style='color:#38bdf8'>File Upload</b> – upload a .wav or .mp3 for REAL Deep Learning inference
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # (Just upload section here)
    st.markdown("""
    <div style='background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:0.8rem;color:#38bdf8'>
      🧠 <b>Real CNN Mode</b> — Upload an audio file for testing offline files.
    </div>
    """, unsafe_allow_html=True)
    
    audio_file = st.file_uploader("Upload audio (.wav, .mp3)", type=["wav", "mp3"], label_visibility="collapsed")
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("🔍 Run Audio AI Inference", use_container_width=True):
            with st.spinner("Processing audio with Librosa and TensorFlow..."):
                try:
                    import tempfile
                    from audio_detection.predict import predict_audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_file.read())
                        res = predict_audio(tmp.name)
                        st.subheader(f"Prediction: {res['class'].upper()} ({res['confidence']*100:.1f}%)")
                except Exception as e:
                    st.error(f"Error: {e}")


def tab_vision():
    st.markdown("""
    <div class='glass-card' style='padding:14px 18px;margin-bottom:4px'>
      <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
        📷 Vision Detection System
      </div>
      <div style='color:#94a3b8;font-size:0.82rem;line-height:1.7'>
        <b style='color:#a78bfa'>Image Upload</b> – upload any photo for REAL YOLOv8 detection<br>
        <b style='color:#34d399'>Webcam Feed</b> – See live stream tab for WebRTC webcam feed!
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.2);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:0.8rem;color:#38bdf8'>
      🧠 <b>Real YOLOv8 Mode</b> — Upload any Image or Video (MP4) to verify architecture offline.
    </div>
    """, unsafe_allow_html=True)

    # Simplified upload form
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
    uploaded = st.file_uploader(
        "Upload media (JPG / PNG / MP4)",
        type=["jpg", "jpeg", "png", "webp", "mp4", "avi"],
        label_visibility="collapsed"
    )
    if uploaded is not None:
        try:
            is_video = uploaded.name.lower().endswith(('.mp4', '.avi', '.mov'))
            if not is_video:
                from PIL import Image
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                if st.button("🔍 Run YOLOv8 Detection", use_container_width=True):
                    with st.spinner("Processing..."):
                        from ultralytics import YOLO
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            img.save(tmp.name, "JPEG")
                            model = YOLO("yolov8m.pt")
                            res = model(tmp.name, conf=conf_thresh, verbose=False)[0]
                            st.image(res.plot()[:, :, ::-1], caption="YOLOv8 Detections", use_container_width=True)
            else:
                st.info("Video processing retained for real files. Use script to process video offline.")
        except Exception as e:
            st.error(f"Error: {e}")


def tab_webrtc():
    st.markdown('''
    <div class='glass-card' style='padding:18px;margin-bottom:14px'>
      <div style='font-size:1.1rem;font-weight:700;color:#34d399;margin-bottom:6px'>🟢 LIVE MULTIMODAL STREAM (NATIVE)</div>
      <div style='color:#94a3b8;font-size:0.85rem;line-height:1.7'>
        This tab accesses your camera securely via Native OpenCV. You can connect a Mobile Phone camera for high-quality visuals!
      </div>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('''
        <div style='font-size:0.8rem; color:#a78bfa; margin-bottom:4px;'><b>Setup Phone Camera:</b> Download "IP Webcam" app on Android/iOS. Open it, click "Start Server", and enter the IPv4 URL below!</div>
        ''', unsafe_allow_html=True)
        
        cam_source = st.text_input("Camera Source URL (Leave blank for default PC Webcam)", placeholder="http://192.168.1.100:8080/video")
        
        use_popout = st.checkbox("🟢 Use Native Pop-Out Window (Fixes ALL Streamlit Lag)", value=True, help="Renders video directly via your GPU in a separate ultra-smooth window rather than pushing through the slow browser dashboard.")
        
        run_cam = st.toggle("🎥 Start Native Camera", value=False)
        FRAME_WINDOW = st.empty()
    with col2:
        INFO_WINDOW = st.empty()

    if run_cam:
        import cv2
        from ultralytics import YOLO
        import numpy as np
        
        try:
            model = YOLO("yolov8m.pt")
            WILDLIFE_MAP = {
                "elephant": ("🐘", (0, 100, 255)),    "bear":     ("🐻", (0, 0, 255)),
                "zebra":    ("🦓", (255, 200, 0)),    "giraffe":  ("🦒", (0, 200, 100)),
                "horse":    ("🐴", (200, 100, 0)),    "cow":      ("🐗", (150, 0, 200)),
                "cat":      ("🐯", (0, 50, 255)),     "dog":      ("🐺", (100, 150, 0)),
            }
            
            source = cam_source.strip() if cam_source.strip() else 0
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                st.error("Cannot access webcam. Check Windows privacy permissions, your mobile IP address, or if another app is using it!")
                return
                
            frame_idx = 0
            # ── AUDIO THREAD SETUP ──
            audio_active = True
            latest_audio_res = None
            try:
                import sounddevice as sd
                from scipy.io.wavfile import write
                import tempfile
                import threading
                from audio_detection.predict import predict_audio
                
                def audio_listener():
                    global latest_audio_res
                    fs = 22050  # Sample rate
                    seconds = 2.5  # Duration
                    while audio_active:
                        try:
                            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
                            sd.wait()
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                write(tmp.name, fs, myrecording)
                                res = predict_audio(tmp.name, use_mock=False)
                                latest_audio_res = res
                                import sys
                                sys.modules['__main__'].GLOBAL_AUDIO_RES = res
                                st.session_state['SHARED_AUDIO_RES'] = res
                        except Exception as e:
                            print(f"Audio Thread Error: {e}")
                            break
                            
                threading.Thread(target=audio_listener, daemon=True).start()
            except ImportError:
                st.warning("Install sounddevice to enable native microphone stream: pip install sounddevice scipy")
                audio_active = False


            
            # Caching variables to keep boxes on screen without stalling CPU
            last_boxes = []
            last_threat = None
            
            if use_popout:
                cv2.namedWindow('WildNode Ultra-Smooth Native Stream', cv2.WINDOW_NORMAL)
            
            while run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost webcam stream connection")
                    break
                    
                frame_idx += 1
                
                # Downscale for performance
                frame = cv2.resize(frame, (640, 480))
                
                # OPTIMIZATION: Heavy YOLO every 5 frames
                if frame_idx % 5 == 0:
                    results = model(frame, conf=0.35, verbose=False)[0]
                    current_boxes = []
                    current_threat = None
                    max_conf = 0.0
                    
                    if results.boxes is not None:
                        for box in results.boxes:
                            cid = int(box.cls[0])
                            cname = results.names[cid].lower()
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            
                            current_boxes.append((cname, conf, x1, y1, x2, y2))
                            
                            if cname in ANIMAL_INFO_DB and conf > max_conf:
                                max_conf = conf
                                current_threat = {'animal': cname, 'confidence': conf}
                    
                    last_boxes = current_boxes
                    if current_threat:
                        last_threat = current_threat
                
                # Draw the cached boxes smoothly every frame
                for b_cname, b_conf, x1, y1, x2, y2 in last_boxes:
                    if b_cname in WILDLIFE_MAP:
                        emoji, color = WILDLIFE_MAP[b_cname]
                        label = f"{b_cname.upper()} {b_conf*100:.0f}%"
                    else:
                        color = (80, 80, 80)
                        label = f"{b_cname} {b_conf*100:.0f}%"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Update info window UI strictly every 5 frames
                if frame_idx % 5 == 0:
                    if last_threat:
                        animal = last_threat['animal']
                        info = ANIMAL_INFO_DB[animal]
                        emoji = info.get('emoji', '🦁')
                        INFO_WINDOW.markdown(f"""
                        <div class='glass-card' style='border-left: 4px solid #38bdf8;'>
                            <h3 style='margin: 0;'>{emoji} {animal.upper()} DETECTED</h3>
                            <div style='color: #a78bfa; font-weight: bold;'>Status: {info['status']} | Diet: {info['diet']}</div>
                            <div style='color: #cbd5e1; margin-top: 8px;'><b>Behavior:</b> {info['behavior']}</div>
                            <div style='color: #ef4444; font-weight: bold; margin-top: 4px;'><b>Action Required:</b> {info['action']}</div>
                            <div style='color: #38bdf8; font-weight: bold; margin-top: 2px;'>Confidence: {last_threat['confidence']*100:.1f}%</div>
                            <div style='color: #fbbf24; font-size:0.75rem; margin-top: 6px;'>Source: {cam_source if cam_source else 'Local Camera'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        INFO_WINDOW.empty()

                if use_popout:
                    cv2.imshow('WildNode Ultra-Smooth Native Stream', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Stream JPEG directly to UI
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        FRAME_WINDOW.image(buffer.tobytes(), use_container_width=True)
                
            audio_active = False
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            st.error(f"Camera Error: {e}")
            if 'cap' in locals(): cap.release()
            cv2.destroyAllWindows()


def tab_about():
    st.markdown("""
    <div class='glass-card' style='padding:28px'>
      <div style='font-size:1.3rem;font-weight:700;color:#38bdf8;margin-bottom:6px'>🌿 WildNode AI</div>
      <div style='color:#64748b;font-size:0.85rem;margin-bottom:16px'>Edge AI-Based Human-Wildlife Conflict Detection System</div>
      <div style='color:#94a3b8;font-size:0.85rem;line-height:1.8'>
        WildNode AI simulates a real-world edge computing node deployed in a wildlife sanctuary.
        It continuously monitors audio and visual feeds, correlates with environmental data,
        and dispatches intelligent alerts to rangers and authorities.
      </div>
    </div>

    <div class='glass-card'>
      <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px'>System Architecture</div>
      <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px'>
        <div style='background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>🔊</div>
          <div style='font-size:0.8rem;font-weight:600;color:#38bdf8;margin:4px 0'>Audio CNN</div>
          <div style='font-size:0.7rem;color:#64748b'>Librosa + TensorFlow</div>
        </div>
        <div style='background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>📷</div>
          <div style='font-size:0.8rem;font-weight:600;color:#a78bfa;margin:4px 0'>Vision YOLO</div>
          <div style='font-size:0.7rem;color:#64748b'>YOLOv8 + OpenCV</div>
        </div>
        <div style='background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>🌦</div>
          <div style='font-size:0.8rem;font-weight:600;color:#34d399;margin:4px 0'>Env Intel</div>
          <div style='font-size:0.7rem;color:#64748b'>OpenWeather API</div>
        </div>
        <div style='background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>🤖</div>
          <div style='font-size:0.8rem;font-weight:600;color:#fbbf24;margin:4px 0'>Decision AI</div>
          <div style='font-size:0.7rem;color:#64748b'>Rule-Based Engine</div>
        </div>
        <div style='background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>📲</div>
          <div style='font-size:0.8rem;font-weight:600;color:#f87171;margin:4px 0'>Alerts</div>
          <div style='font-size:0.7rem;color:#64748b'>WhatsApp + SMS</div>
        </div>
        <div style='background:rgba(56,189,248,0.06);border:1px solid rgba(56,189,248,0.15);border-radius:10px;padding:14px;text-align:center'>
          <div style='font-size:1.5rem'>📊</div>
          <div style='font-size:0.8rem;font-weight:600;color:#38bdf8;margin:4px 0'>Dashboard</div>
          <div style='font-size:0.7rem;color:#64748b'>Streamlit + Plotly</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN RENDER LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    refresh_rate = render_sidebar()

    # Header
    ist_now = datetime.now(IST).strftime("%d %b %Y · %I:%M:%S %p IST")
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:10px'>
      <div>
        <h1 style='margin:0;font-size:1.7rem;background:linear-gradient(135deg,#38bdf8,#a78bfa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800'>
          🌿 WildNode AI Monitor
        </h1>
        <p style='margin:2px 0 0;font-size:0.75rem;color:#334155'>
          Edge AI · Human-Wildlife Conflict Detection · {st.session_state.sector}
        </p>
      </div>
      <div style='font-size:0.75rem;color:#475569;text-align:right'>
        🕐 {ist_now}<br>
        <span style='color:#334155'>Cycle #{st.session_state.cycle}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["📊 Overview", "🔴 Live Stream", "🔊 Audio", "📷 Vision", "🚨 Alerts", "ℹ️ About"])

    with tabs[0]: tab_overview()
    with tabs[1]: tab_webrtc()
    with tabs[2]: tab_audio()
    with tabs[3]: tab_vision()
    with tabs[4]: tab_alerts()
    with tabs[5]: tab_about()

    # Display ANIMAL INFO for latest detection
    if st.session_state.last_decision and st.session_state.last_decision.get("threats_detected"):
        threats = st.session_state.last_decision["threats_detected"]
        for threat in threats:
            animal = threat["animal"]
            if animal in ANIMAL_INFO_DB:
                info = ANIMAL_INFO_DB[animal]
                emoji = info.get('emoji', '🦁')
                st.markdown(f"""
                <div class='glass-card' style='border-left: 4px solid #38bdf8;'>
                    <h3 style='margin: 0;'>{emoji} {animal.upper()} DETECTED - LIVE INTELLIGENCE</h3>
                    <div style='color: #a78bfa; font-weight: bold;'>Status: {info['status']} | Diet: {info['diet']}</div>
                    <div style='color: #cbd5e1; margin-top: 8px;'><b>Behavior:</b> {info['behavior']}</div>
                    <div style='color: #ef4444; font-weight: bold; margin-top: 4px;'><b>Action Required:</b> {info['action']}</div>
                </div>
                """, unsafe_allow_html=True)
                break # Just show top priority one for space



if __name__ == "__main__":
    main()
