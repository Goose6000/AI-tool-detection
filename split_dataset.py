"""Split labeled data into train/val in YOLO layout.

Expects images and matching YOLO .txt labels together in a source folder
(e.g. what Label Studio / CVAT / labelImg exports), then writes:
  dataset/images/{train,val}   dataset/labels/{train,val}

Usage:  python split_dataset.py --src dataset/labeled --val 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/labeled")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    images = [p for p in images if p.with_suffix(".txt").exists()]
    if not images:
        raise SystemExit("No image + .txt label pairs found in " + str(src))

    random.Random(args.seed).shuffle(images)
    n_val = max(1, int(len(images) * args.val))
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, dst / "images" / split / img.name)
            shutil.copy2(img.with_suffix(".txt"), dst / "labels" / split / (img.stem + ".txt"))
        print(f"{split}: {len(files)} images")


if __name__ == "__main__":
    main()
