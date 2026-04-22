"""
wildnode_ai/vision_detection/webcam_detect.py
===============================================
PURPOSE: Run REAL YOLOv8 detection on your built-in laptop/PC webcam.
         Streams live frames and shows bounding boxes in real time.

HOW TO RUN:
  python vision_detection/webcam_detect.py              # Default camera (index 0)
  python vision_detection/webcam_detect.py --camera 1   # Second camera

CONTROLS (in the popup window):
  Q or ESC → Quit

REQUIREMENTS:
  pip install ultralytics opencv-python

BEGINNER NOTE:
  YOLOv8 runs on every frame from your webcam and draws colored boxes
  around anything it detects. It knows 80 object types from COCO training.
  Animals like elephant, bear, zebra, giraffe are included!
  For indoor testing: it will detect people, chairs, phones etc.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_webcam_detection(camera_index: int = 0,
                         conf_threshold: float = 0.40,
                         max_frames: int = None):
    """
    Run real-time YOLOv8 detection from webcam.

    Args:
        camera_index   : 0 = default webcam, 1 = second camera
        conf_threshold : Min confidence for showing detection
        max_frames     : Stop after N frames (None = run until Q pressed)
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Missing dependencies! Run:\n  pip install ultralytics opencv-python")
        return

    # Wildlife class mapping
    WILDLIFE_MAP = {
        "elephant": ("🐘", (0, 100, 255)),    # Orange
        "bear":     ("🐻", (0, 0, 255)),      # Red
        "zebra":    ("🦓", (255, 200, 0)),    # Cyan
        "giraffe":  ("🦒", (0, 200, 100)),    # Green
        "horse":    ("🐴", (200, 100, 0)),    # Blue
        "cow":      ("🐗", (150, 0, 200)),    # Purple
        "cat":      ("🐯", (0, 50, 255)),     # Deep orange (tiger proxy)
        "dog":      ("🐺", (100, 150, 0)),    # Olive
    }
    ALL_CLASSES = set(WILDLIFE_MAP.keys())

    print("[INFO] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"[INFO] Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}.")
        print("  Make sure your webcam is connected and not used by another app.")
        return

    # Get camera properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Camera: {width}x{height} @ {fps:.0f}fps")
    print("[INFO] Press Q or ESC to quit\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        frame_count += 1

        # Run YOLOv8 inference
        results = model(frame, conf=conf_threshold, verbose=False)
        result  = results[0]

        # Draw detections on frame
        wildlife_found = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id  = int(box.cls[0])
                coco_name = result.names[class_id].lower()
                conf      = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                if coco_name in ALL_CLASSES:
                    emoji, color = WILDLIFE_MAP[coco_name]
                    label = f"{coco_name.upper()} {conf*100:.0f}%"
                    wildlife_found.append(coco_name)
                else:
                    color = (80, 80, 80)
                    label = f"{coco_name} {conf*100:.0f}%"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay: WildNode header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 36), (10, 20, 40), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        status = f"  WildNode AI | Frame #{frame_count} | Wildlife: {', '.join(wildlife_found) or 'None'}"
        cv2.putText(frame, status, (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)

        cv2.imshow("WildNode AI – Live Detection (Q to quit)", frame)

        # Print console output for wildlife
        if wildlife_found:
            print(f"[Frame {frame_count:04d}] 🚨 WILDLIFE DETECTED: {wildlife_found}")

        # Exit on Q or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[✓] Webcam session ended. Total frames processed: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildNode AI – Webcam Real-Time Detector")
    parser.add_argument("--camera", type=int,   default=0,    help="Camera index (default: 0)")
    parser.add_argument("--conf",   type=float, default=0.40, help="Confidence threshold")
    args = parser.parse_args()

    run_webcam_detection(camera_index=args.camera, conf_threshold=args.conf)
