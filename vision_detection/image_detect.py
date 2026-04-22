"""
wildnode_ai/vision_detection/image_detect.py
============================================
PURPOSE: Run REAL YOLOv8 detection on a single image file.
         This is NOT simulation — it uses actual neural network inference.

HOW TO RUN:
  python vision_detection/image_detect.py --image path/to/elephant.jpg
  python vision_detection/image_detect.py --image path/to/photo.jpg --show

REQUIREMENTS:
  pip install ultralytics opencv-python pillow

OUTPUT:
  - Prints detections with class name, confidence, bounding box
  - Optionally saves an annotated image with boxes drawn
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_real_detection(image_path: str,
                       save_output: bool = True,
                       conf_threshold: float = 0.35) -> list:
    """
    Run real YOLOv8 inference on an image file.

    Args:
        image_path     : Path to image (.jpg / .png / .webp)
        save_output    : Save annotated image with bounding boxes
        conf_threshold : Minimum confidence to show a detection

    Returns:
        detections : List of dicts (same format as simulate_vision_detection)
    """
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
    except ImportError:
        print("[ERROR] Missing dependencies! Run:\n  pip install ultralytics opencv-python")
        return []

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return []

    print(f"[INFO] Running YOLOv8 on: {image_path}")
    print(f"[INFO] Confidence threshold: {conf_threshold}")

    # Load model (downloads yolov8n.pt ~6MB on first run)
    model = YOLO("yolov8n.pt")

    # Run inference
    results = model(image_path, conf=conf_threshold, verbose=False)

    # Save annotated image
    if save_output:
        out_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
        results[0].save(filename=out_path)
        print(f"[✓] Annotated image saved: {out_path}")

    # --- COCO class → Wildlife label mapping ---
    WILDLIFE_MAP = {
        "elephant": "🐘 elephant",
        "bear":     "🐻 bear",
        "zebra":    "🦓 zebra",
        "giraffe":  "🦒 giraffe",
        "horse":    "🐴 horse",
        "cow":      "🐗 wild boar",
        "cat":      "🐯 tiger",
        "dog":      "🐺 wild dog",
    }

    HIGH_PRIORITY = {"elephant", "tiger", "bear", "wild boar"}

    detections = []
    result = results[0]

    # If no boxes found
    if result.boxes is None or len(result.boxes) == 0:
        print("\n[RESULT] No wildlife detected in this image.")
        return []

    print(f"\n[RESULT] Found {len(result.boxes)} object(s):\n")

    for box in result.boxes:
        class_id   = int(box.cls[0])
        coco_name  = result.names[class_id].lower()
        confidence = float(box.conf[0])
        bbox       = [int(v) for v in box.xyxy[0].tolist()]

        # Map to wildlife (or keep coco class)
        if coco_name in WILDLIFE_MAP:
            label = WILDLIFE_MAP[coco_name]
            wildlife_name = label.split(" ", 1)[1]
        else:
            label = f"🔍 {coco_name}"
            wildlife_name = coco_name

        priority = "HIGH" if wildlife_name in HIGH_PRIORITY else "MEDIUM"

        det = {
            "class"      : wildlife_name,
            "coco_class" : coco_name,
            "label"      : f"{label} ({confidence*100:.1f}%)",
            "confidence" : confidence,
            "bbox"       : bbox,
            "priority"   : priority,
        }
        detections.append(det)

        print(f"  {'🚨' if priority=='HIGH' else '📡'} {label}")
        print(f"     Confidence : {confidence*100:.1f}%")
        print(f"     Priority   : {priority}")
        print(f"     Bounding box: x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}")
        print()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildNode AI – Real Image Detector")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--conf",  type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--show",  action="store_true", help="Display image window")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated image")
    args = parser.parse_args()

    detections = run_real_detection(
        args.image,
        save_output=not args.no_save,
        conf_threshold=args.conf
    )

    if args.show and detections:
        try:
            import cv2
            img = cv2.imread(args.image)
            cv2.imshow("WildNode AI – Detection", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[WARN] Cannot show image: {e}")
