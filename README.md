# People Counter — YOLO11
Real-time people detection and per-frame counting using Ultralytics YOLO11. Process webcam or video files, visualize detections, and save annotated output.

---

## Features
- Detects people (COCO 'person' class) using Ultralytics YOLO11 models.
- Live webcam support or video file input.
- Per-frame people count overlay and FPS display.
- Optional output video recording (annotated).
- Robust for headless servers (no GUI) with automatic fallback to saving video + console logging.
- Easy to extend with tracking (DeepSORT) or entry/exit counters.

---

## Demo
Add a short GIF or MP4 here showing webcam/video with bounding boxes and the people counter overlay. (Example: `demo/demo.gif`)

---

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
