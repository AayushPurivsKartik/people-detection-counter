#!/usr/bin/env python3
"""
people_counter_fixed.py

Usage:
  python people_counter_fixed.py                # webcam (default), shows window
  python people_counter_fixed.py --source video.mp4     # file, shows window
  python people_counter_fixed.py --no-show --output out.mp4   # don't show, save annotated video
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

from ultralytics import YOLO

PERSON_CLASS_ID = 0

def parse_args():
    p = argparse.ArgumentParser(description="YOLO11 People Detector + Counter (improved)")
    p.add_argument("--source", "-s", default="0",
                   help="Video source: integer (0,1,..) for webcam or path to file. Default=0")
    p.add_argument("--model", "-m", default="yolo11n.pt",
                   help="YOLO11 model path or name (default: yolo11n.pt)")
    p.add_argument("--conf", "-c", type=float, default=0.25,
                   help="Confidence threshold (default 0.25)")
    p.add_argument("--output", "-o", default=None,
                   help="Optional path to save processed video (e.g. out.mp4)")
    # show by default; allow --no-show to turn off windows
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Do not display GUI windows (useful for headless servers)")
    p.set_defaults(show=True)
    return p.parse_args()

def open_capture(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {src}")
    return cap

def try_convert_to_numpy(x):
    try:
        return x.cpu().numpy()
    except Exception:
        try:
            return np.asarray(x)
        except Exception:
            return None

def extract_person_boxes(res, conf_thres):
    person_boxes = []
    try:
        # Preferred: res.boxes.xyxy, .cls, .conf
        xyxy = try_convert_to_numpy(res.boxes.xyxy)
        cls  = try_convert_to_numpy(res.boxes.cls)
        conf = try_convert_to_numpy(res.boxes.conf)
        if xyxy is not None and cls is not None:
            for b, c, cf in zip(xyxy, cls, conf):
                if int(c) == PERSON_CLASS_ID and float(cf) >= conf_thres:
                    x1, y1, x2, y2 = b
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2), float(cf)))
            return person_boxes
    except Exception:
        pass

    # Fallback: res.boxes.data (xyxy, conf, cls)
    try:
        data = try_convert_to_numpy(res.boxes.data)
        if data is not None:
            for d in data:
                x1, y1, x2, y2, cf, cl = d
                if int(cl) == PERSON_CLASS_ID and float(cf) >= conf_thres:
                    person_boxes.append((int(x1), int(y1), int(x2), int(y2), float(cf)))
            return person_boxes
    except Exception:
        pass

    return person_boxes

def main():
    args = parse_args()
    print("[INFO] Starting people_counter_fixed.py")
    print(f"[INFO] Source: {args.source}  Model: {args.model}  Conf: {args.conf}  Show window: {args.show}  Output: {args.output}")

    # Detect if there is a display available (Linux DISPLAY or Windows)
    has_display = False
    if os.name == "nt":
        has_display = True
    else:
        if os.environ.get("DISPLAY"):
            has_display = True

    if args.show and not has_display:
        print("[WARNING] No DISPLAY found. Disabling GUI display automatically.")
        args.show = False

    try:
        cap = open_capture(args.source)
    except Exception as e:
        print("[ERROR] Could not open source:", e)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    print(f"[INFO] Video opened: {w}x{h} @ {fps:.2f} FPS")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("[WARNING] Could not open video writer. Output will not be saved.")
            writer = None
        else:
            print(f"[INFO] Writing output to {args.output}")

    # Load model
    try:
        model = YOLO(args.model)
        print("[INFO] YOLO model loaded.")
    except Exception as e:
        print("[ERROR] Failed to load YOLO model:", e)
        cap.release()
        sys.exit(1)

    if args.show:
        window_name = "People Counter (press q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    start_time = time.time()
    last_print = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream or can't read frame.")
                break
            frame_idx += 1

            # Ultralytics expects RGB in many cases; convert to RGB
            rgb = frame[..., ::-1]

            # Run inference
            try:
                results = model(rgb)   # pass numpy RGB image
                res = results[0]
            except Exception as e:
                print("[ERROR] Inference failed on frame", frame_idx, ":", e)
                break

            person_boxes = extract_person_boxes(res, args.conf)

            # Draw boxes
            for (x1, y1, x2, y2, cf) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"person {cf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw, y1), (0, 200, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            count = len(person_boxes)
            elapsed = max(1e-6, time.time() - start_time)
            cur_fps = frame_idx / elapsed
            info = f"People: {count}   FPS: {cur_fps:.1f}"
            cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Show or save
            if args.show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Quit pressed.")
                    break

            if writer:
                writer.write(frame)

            # Print counts occasionally (every 30 frames) so console isn't spammed
            if frame_idx - last_print >= 30:
                print(f"[INFO] Frame {frame_idx}: People={count}  FPS={cur_fps:.1f}")
                last_print = frame_idx

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        print("[INFO] Exiting.")

if __name__ == "__main__":
    main()
