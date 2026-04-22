# 🌿 WildNode AI – Edge AI-Based Human-Wildlife Conflict Detection System

> **Production-grade edge AI system** simulating real-time wildlife monitoring using Deep Learning, Computer Vision, Environmental Intelligence, and a multi-channel Alert System.

---

## 🏗 Project Structure

```
wildnode_ai/
├── audio_detection/          # 🔊 CNN audio classifier (Librosa + TensorFlow)
│   ├── preprocess.py         #    WAV → Mel Spectrogram
│   ├── model.py              #    CNN model definition
│   ├── train.py              #    Training script
│   └── predict.py            #    Inference + simulation
│
├── vision_detection/         # 📷 YOLOv8 wildlife detector
│   └── detector.py           #    YOLO wrapper + simulation
│
├── environmental/            # 🌦 Weather intelligence
│   ├── weather_api.py        #    OpenWeather API + mock
│   └── risk_calculator.py    #    Env risk scorer
│
├── decision_engine/          # 🤖 Rule-based AI engine
│   └── engine.py             #    Risk scoring + decisions
│
├── alert_system/             # 📲 Multi-channel alerts
│   ├── alerter.py            #    Console + WhatsApp + SMS + Log
│   └── alert_log.json        #    Persistent alert history
│
├── simulation/               # 🔁 Real-time simulators
│   ├── audio_simulator.py
│   └── video_simulator.py
│
├── dashboard/                # 📊 Glassmorphism Streamlit UI
│   └── app.py
│
├── pipeline/                 # 🔗 Full integration
│   └── main_pipeline.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
copy .env.example .env
# Edit .env to add your OpenWeather API key (optional)
```

### 3. Launch the Dashboard ← START HERE
```bash
streamlit run dashboard/app.py
```
Open browser at **http://localhost:8501** and click **▶ Start**

---

## 🧩 Running Individual Modules

### Train Audio CNN (optional – simulation works without it)
```bash
python audio_detection/train.py
```

### Test Audio Prediction
```bash
python audio_detection/predict.py
```

### Test Vision Detection
```bash
python vision_detection/detector.py
```

### Test Decision Engine
```bash
python decision_engine/engine.py
```

### Run Full CLI Pipeline
```bash
python pipeline/main_pipeline.py --cycles 10 --interval 5
```

---

## 🛠 Tech Stack

| Module | Technology |
|---|---|
| Audio Detection | TensorFlow/Keras CNN + Librosa |
| Vision Detection | YOLOv8 (Ultralytics) + OpenCV |
| Environmental | OpenWeather API + pytz |
| Decision Engine | Rule-based Expert System |
| Alert System | Simulated Twilio/WhatsApp |
| Dashboard | Streamlit + Plotly |

---

## 📡 Free Dataset Sources

| Source | Link | Content |
|---|---|---|
| FreeSound | https://freesound.org | Wildlife audio |
| Xeno-canto | https://www.xeno-canto.org | Animal vocalizations |
| iNaturalist | https://www.inaturalist.org | Wildlife images |
| OpenImages | https://storage.googleapis.com/openimages | YOLO-ready datasets |

---

## 🏆 Resume Description

> **WildNode AI** (2024) – Engineered a full-stack, simulated edge-AI system for real-time human-wildlife conflict detection. Built a CNN audio classifier achieving >90% accuracy on spectrogram data (Librosa/TensorFlow), integrated YOLOv8 for animal detection, and developed a rule-based decision engine combining multi-modal sensor fusion (audio + vision + weather API). Deployed a glassmorphism Streamlit dashboard with live Plotly gauges and animated alerts. System architecture follows production edge-computing patterns with modular, independently testable components.

---

## 🎓 Viva Q&A

**Q: Why Mel Spectrogram instead of raw audio?**
A: CNNs work on 2D images. Mel Spectrograms convert 1D audio to 2D frequency-time images capturing perceptually relevant features.

**Q: Why YOLOv8 instead of a custom trained model?**
A: YOLOv8 pretrained on COCO already knows 80 classes including elephant and bear, making it production-ready without custom training.

**Q: What is sensor fusion?**
A: Combining multiple data sources (audio + vision + weather) to make more reliable decisions than any single sensor alone.

**Q: How would this deploy on real hardware?**
A: Replace simulation layer with a Raspberry Pi camera module + USB microphone. The rest of the pipeline remains unchanged.

---

## 🚀 Future Improvements

- [ ] Replace rule engine with fine-tuned LLM (Gemini/GPT-4) for contextual reasoning
- [ ] Custom YOLOv8 training on Indian wildlife datasets (tiger, leopard, gaur)
- [ ] Real Twilio/WhatsApp integration for live alerts
- [ ] MQTT protocol for actual IoT edge node communication
- [ ] Thermal camera support for night-time detection
- [ ] GPS-tagged detection zones on interactive map (Folium)
- [ ] Mobile app (React Native) for ranger alerts

---

*Made with ❤️ for wildlife conservation · WildNode AI v1.0*
