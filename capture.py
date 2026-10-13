"""Capture training photos from a camera.

SPACE = save frame, Q = quit.
Frames go to dataset/raw/<label>/ so you can label them afterwards.

Usage:  python capture.py --label hammer --source 0
"""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="folder name for this batch, e.g. hammer or drill_with_battery")
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", default="dataset/raw")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/source {args.source}")

    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    count = len(list(out.glob("*.jpg")))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"{args.label}  saved: {count}  [SPACE=save Q=quit]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(str(out / f"{args.label}_{int(time.time() * 1000)}.jpg"), frame)
            count += 1
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
