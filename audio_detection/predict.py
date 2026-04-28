"""
wildnode_ai/audio_detection/predict.py
========================================
PURPOSE: Load a trained CNN and predict the animal class from a .wav file.
         Returns the class name and confidence score.

HOW TO RUN:
  python audio_detection/predict.py --audio audio_detection/samples/elephant_demo.wav

OUTPUT:
  {
    "class": "elephant",
    "confidence": 0.94,
    "all_scores": {"elephant": 0.94, "tiger": 0.04, "background": 0.02}
  }
"""

import os
import sys
import json
import argparse
import numpy as np

# ─── Handle imports whether run standalone or as module ──────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "audio_model.keras")

# ─── CLASS_NAMES defined here (no TF needed) ─────────────────────────────────
CLASS_NAMES = ["elephant", "wild_boar", "background"]

# ─────────────────────────────────────────────────────────────────────────────

# Lazy-load model (TF only imported when actually running inference)
_model = None


def get_model():
    """Load the trained model from disk (cached after first load)."""
    global _model
    if _model is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at: {MODEL_PATH}\n"
                f"Please run: python audio_detection/train.py  first!"
            )
        print(f"[INFO] Loading audio model from: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_audio(file_path: str, use_mock: bool = False) -> dict:
    """
    Predict the wildlife class from a .wav audio file.

    Args:
        file_path : Path to the .wav file to analyze
        use_mock  : If True, use a random prediction (no model needed)

    Returns:
        result : Dictionary with keys:
                   "class"      → Predicted animal name (str)
                   "confidence" → Probability of top prediction (float 0–1)
                   "all_scores" → Dict of all class probabilities
                   "file"       → Input file path
    """
    from audio_detection.preprocess import wav_to_spectrogram_array
    # CLASS_NAMES is defined at module level (no TF import needed)

    if use_mock:
        # ── MOCK MODE: Generate fake random prediction ─────────────────
        # Useful when model isn't trained yet or for demos
        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))   # Random probabilities that sum to 1
        pred_idx = int(np.argmax(probs))
        return {
            "class"      : CLASS_NAMES[pred_idx],
            "confidence" : float(probs[pred_idx]),
            "all_scores" : {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "file"       : file_path,
            "mode"       : "mock"
        }

    # ── REAL MODEL MODE ────────────────────────────────────────────────
    # Step 1: Convert .wav to mel spectrogram
    mel = wav_to_spectrogram_array(file_path)   # Shape: (128, 216)

    # Step 2: Resize to model input: (128, 216, 1) → add batch dim → (1, 128, 216, 1)
    target_w = 216
    if mel.shape[1] < target_w:
        mel = np.pad(mel, ((0, 0), (0, target_w - mel.shape[1])))
    else:
        mel = mel[:, :target_w]

    X = mel[np.newaxis, :, :, np.newaxis].astype(np.float32)   # (1, 128, 216, 1)

    # Step 3: Run inference
    model  = get_model()
    probs  = model.predict(X, verbose=0)[0]   # Shape: (3,)

    # Step 4: Parse results
    pred_idx = int(np.argmax(probs))
    return {
        "class"      : CLASS_NAMES[pred_idx],
        "confidence" : float(probs[pred_idx]),
        "all_scores" : {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "file"       : file_path,
        "mode"       : "model"
    }


def simulate_audio_detection():
    """
    Run a simulated audio detection cycle over all sample files.
    Used by the simulation engine and dashboard.

    Returns:
        result : Prediction dict (same format as predict_audio)
    """
    import random
    # CLASS_NAMES defined at module level

    # Weighted probabilities — elephant and wild boar are crop threats
    weights = [0.40, 0.40, 0.20]
    chosen_class = random.choices(CLASS_NAMES, weights=weights, k=1)[0]
    class_idx = CLASS_NAMES.index(chosen_class)

    # Generate realistic-looking probabilities
    raw = np.zeros(len(CLASS_NAMES))
    raw[class_idx] = np.random.uniform(0.60, 0.96)
    remaining = 1 - raw[class_idx]
    other_idxs = [i for i in range(len(CLASS_NAMES)) if i != class_idx]
    split = np.random.uniform(0, remaining)
    raw[other_idxs[0]] = split
    raw[other_idxs[1]] = remaining - split

    return {
        "class"      : chosen_class,
        "confidence" : float(raw[class_idx]),
        "all_scores" : {c: float(raw[i]) for i, c in enumerate(CLASS_NAMES)},
        "file"       : "simulated_stream",
        "mode"       : "simulation"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildNode AI – Audio Predictor")
    parser.add_argument("--audio", type=str, help="Path to .wav audio file")
    parser.add_argument("--mock",  action="store_true",
                        help="Use mock prediction (no model required)")
    args = parser.parse_args()

    if not args.audio and not args.mock:
        print("[INFO] No audio file given. Running in simulation mode...\n")
        result = simulate_audio_detection()
    elif args.mock:
        result = predict_audio(args.audio or "demo.wav", use_mock=True)
    else:
        result = predict_audio(args.audio)

    # ── Pretty Print Result ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  🔊 WildNode AI – Audio Detection Result")
    print("=" * 50)
    emoji_map = {"elephant": "🐘", "wild_boar": "🐗", "background": "🌿"}
    emoji = emoji_map.get(result["class"], "❓")
    print(f"  Detected   : {emoji}  {result['class'].upper()}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"  Mode       : {result.get('mode', 'unknown')}")
    print(f"\n  All Scores:")
    for cls, score in result["all_scores"].items():
        bar = "█" * int(score * 20)
        print(f"    {cls:<12}: {bar:<20} {score:.3f}")
    print("=" * 50)

    print(f"\n  JSON Output:\n{json.dumps(result, indent=2)}")
