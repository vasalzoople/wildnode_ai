"""
wildnode_ai/vision_detection/detector.py
==========================================
PURPOSE: YOLOv8 wrapper for detecting wildlife in images/video frames.

HOW IT WORKS:
  1. Loads a pretrained YOLOv8 model (trained on COCO dataset)
  2. COCO dataset already contains: elephant, bear, zebra, giraffe, horse, cow, dog
  3. We map these to our wildlife categories
  4. Runs inference → returns bounding boxes + confidence for each detection

BEGINNER NOTE:
  YOLOv8 is a state-of-the-art object detection model.
  "Pretrained on COCO" means it was trained on 80 common object categories
  and already knows what elephants look like — no extra training needed!
  We just filter the results for wildlife-relevant classes.

REQUIREMENTS:
  pip install ultralytics opencv-python
"""

import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ─── Wildlife Class Mapping ───────────────────────────────────────────────────
# COCO class names → our WildNode wildlife labels
# YOLOv8 can detect these COCO classes natively
WILDLIFE_CLASS_MAP = {
    "elephant" : "elephant",   # COCO class 61
    "bear"     : "bear",       # COCO class 23
    "zebra"    : "zebra",      # COCO class 24
    "giraffe"  : "giraffe",    # COCO class 25
    "horse"    : "horse",      # Treated as large wildlife
    "cow"      : "wild boar",  # Approximate mapping for demo
    "dog"      : "wild dog",   # Approximate for canids
    "cat"      : "tiger",      # Approximate for felines in demo
}

# Classes we actively alert on (higher priority)
HIGH_PRIORITY_CLASSES = {"elephant", "tiger", "bear", "wild boar"}

# ─────────────────────────────────────────────────────────────────────────────

_model = None   # Cached YOLO model


def get_yolo_model(model_name: str = "yolov8n.pt"):
    """
    Load and cache the YOLOv8 model. Downloads automatically on first run.

    Args:
        model_name : YOLOv8 variant. Options:
                       yolov8n.pt  → Nano  (fastest, ~3MB)
                       yolov8s.pt  → Small (~11MB)
                       yolov8m.pt  → Medium (best accuracy for production)

    Returns:
        model : Loaded YOLO model object
    """
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
            print(f"[INFO] Loading YOLOv8 model: {model_name}")
            _model = YOLO(model_name)
            print("[✓] YOLOv8 model loaded successfully!")
        except ImportError:
            print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            return None
    return _model


def detect_wildlife(source, conf_threshold: float = 0.45,
                    model_name: str = "yolov8n.pt") -> list:
    """
    Run YOLOv8 detection on an image, video frame, or file path.

    Args:
        source         : Can be:
                           - str (image/video file path)
                           - numpy array (BGR frame from OpenCV)
                           - int (webcam index, e.g., 0)
        conf_threshold : Minimum confidence to include a detection
        model_name     : Which YOLOv8 variant to use

    Returns:
        detections : List of dicts, each containing:
                       {
                         "class"      : "elephant",
                         "label"      : "🐘 elephant (94.2%)",
                         "confidence" : 0.942,
                         "bbox"       : [x1, y1, x2, y2],  # Pixel coords
                         "priority"   : "HIGH",
                       }
    """
    model = get_yolo_model(model_name)
    if model is None:
        return []

    try:
        # Run YOLO inference
        results = model(source, conf=conf_threshold, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Get class name from COCO dataset
                class_id   = int(box.cls[0])
                coco_name  = result.names[class_id]
                confidence = float(box.conf[0])

                # Check if it's a wildlife-relevant class
                if coco_name.lower() not in WILDLIFE_CLASS_MAP:
                    continue

                wildlife_name = WILDLIFE_CLASS_MAP[coco_name.lower()]
                bbox = box.xyxy[0].tolist()   # [x1, y1, x2, y2]

                emoji_map = {
                    "elephant"  : "🐘",
                    "tiger"     : "🐯",
                    "bear"      : "🐻",
                    "wild boar" : "🐗",
                    "zebra"     : "🦓",
                    "giraffe"   : "🦒",
                    "horse"     : "🐴",
                    "wild dog"  : "🐺",
                }
                emoji = emoji_map.get(wildlife_name, "🦁")

                detections.append({
                    "class"      : wildlife_name,
                    "coco_class" : coco_name,
                    "label"      : f"{emoji} {wildlife_name} ({confidence*100:.1f}%)",
                    "confidence" : confidence,
                    "bbox"       : [int(v) for v in bbox],
                    "priority"   : "HIGH" if wildlife_name in HIGH_PRIORITY_CLASSES else "MEDIUM",
                })

        return detections

    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        return []


def simulate_vision_detection() -> list:
    """
    Generate a realistic simulated vision detection result.
    Used by the dashboard and simulation engine when no camera is available.

    Returns:
        detections : List of detection dicts (same format as detect_wildlife)
    """
    animals = [
        {"class": "elephant", "emoji": "🐘", "priority": "HIGH"},
        {"class": "tiger",    "emoji": "🐯", "priority": "HIGH"},
        {"class": "wild boar","emoji": "🐗", "priority": "HIGH"},
        {"class": "bear",     "emoji": "🐻", "priority": "HIGH"},
        {"class": "zebra",    "emoji": "🦓", "priority": "MEDIUM"},
    ]

    # Randomly decide: 0 detections (30%), 1 detection (50%), 2 detections (20%)
    n_detections = random.choices([0, 1, 2], weights=[30, 50, 20])[0]
    if n_detections == 0:
        return []

    chosen = random.sample(animals, min(n_detections, len(animals)))
    detections = []

    for animal in chosen:
        confidence = round(random.uniform(0.58, 0.97), 3)
        # Random bbox in a 640×480 frame
        x1, y1 = random.randint(50, 300), random.randint(50, 200)
        x2, y2 = x1 + random.randint(80, 200), y1 + random.randint(80, 200)

        detections.append({
            "class"      : animal["class"],
            "coco_class" : animal["class"],
            "label"      : f"{animal['emoji']} {animal['class']} ({confidence*100:.1f}%)",
            "confidence" : confidence,
            "bbox"       : [x1, y1, x2, y2],
            "priority"   : animal["priority"],
            "mode"       : "simulation"
        })

    return detections


if __name__ == "__main__":
    print("WildNode AI – Vision Simulator Test")
    results = simulate_vision_detection()
    if results:
        for r in results:
            print(f"  ✓ {r['label']}  | Priority: {r['priority']} | BBox: {r['bbox']}")
    else:
        print("  No wildlife detected (background/clear frame)")
