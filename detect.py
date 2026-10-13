"""Live tool detection with battery-in-tool check and removed-tool tracking.

Usage:
  python detect.py --weights runs/detect/tools/weights/best.pt --source 0

Keys: Q = quit, R = reset inventory baseline to what is currently visible.
Events are printed and appended to events.jsonl (fully automatic, no clicks needed).
"""
import argparse
import json
import time

import cv2
from ultralytics import YOLO

from logic import BATTERY, CLASS_NAMES, DRILL, Det, Inventory, Stabilizer, battery_status

PALETTE = [(255, 99, 71), (60, 179, 113), (30, 144, 255), (255, 165, 0), (186, 85, 211),
           (0, 206, 209), (255, 215, 0), (154, 205, 50), (220, 20, 60), (0, 255, 127)]
COLORS = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(CLASS_NAMES)}


def log_event(path, kind, **data):
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **data}
    print(json.dumps(rec))
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/tools/weights/best.pt")
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-cover", type=float, default=0.6, help="battery overlap with drill to count as installed")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--need", type=int, default=8)
    ap.add_argument("--log", default="events.jsonl")
    ap.add_argument("--no-window", action="store_true", help="headless mode, log only")
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source}")

    stab = Stabilizer(args.window, args.need)
    inv = Inventory()
    last_battery_state = None

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue

        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            name = model.names[int(b.cls)]
            dets.append(Det(name, float(b.conf), tuple(b.xyxy[0].tolist())))

        present = stab.update(d.name for d in dets)
        for kind, name in inv.update(present):
            log_event(args.log, kind, tool=name, missing=inv.missing())

        bs = battery_status(dets, args.min_cover)
        if DRILL in present or BATTERY in present:
            state = (bs["installed"], bs["loose"])
            if state != last_battery_state:
                if bs["installed"]:
                    log_event(args.log, "BATTERY_IN_TOOL", **bs)
                elif bs["batteries_total"]:
                    log_event(args.log, "BATTERY_LOOSE", **bs)
                last_battery_state = state

        if args.no_window:
            continue

        for d in dets:
            x1, y1, x2, y2 = map(int, d.box)
            c = COLORS.get(d.name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{d.name} {d.conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

        missing = inv.missing()
        lines = [f"Battery in tool: {bs['installed']}  loose: {bs['loose']}",
                 "Missing: " + (", ".join(missing) if missing else "none")]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("tool detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            inv.state = set(present)
            log_event(args.log, "BASELINE_RESET", present=sorted(present))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
