"""Merge a downloaded Roboflow (YOLOv8-format) dataset into our dataset/ folder,
remapping its class names to ours (see data.yaml). Unmapped classes are dropped.

Step 1 - see what classes the download has:
  python merge_datasets.py --src ~/Downloads/workshop-tools --list

Step 2 - merge, mapping THEIR names to OUR names (their=our, comma separated):
  python merge_datasets.py --src ~/Downloads/workshop-tools --prefix ws \
      --map "Hammer=hammer,Pliers=pliers,Screwdriver=screwdriver,Adjustable Wrench=adjustable_spanner"

Run once per downloaded dataset, with a different --prefix each time.
Matching of names ignores case, spaces, dashes and underscores.
Images with no mapped objects are skipped (so the model is not taught that a
real tool is background). Roboflow 'test' images go into train.
"""
import argparse
import shutil
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def load_names(yaml_path):
    names = yaml.safe_load(Path(yaml_path).read_text())["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def to_bbox(vals):
    """YOLO box (4 values) or polygon (>=6 values) -> cx, cy, w, h."""
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted Roboflow folder (contains data.yaml)")
    ap.add_argument("--ours", default="data.yaml")
    ap.add_argument("--dst", default="dataset")
    ap.add_argument("--map", default="", help='"their=our,their2=our2"')
    ap.add_argument("--prefix", default="ext", help="added to filenames to avoid collisions")
    ap.add_argument("--list", action="store_true", help="only print the dataset's classes")
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst)
    their = load_names(src / "data.yaml")
    ours = load_names(args.ours)

    if args.list:
        print("Classes in this dataset:")
        for i, n in their.items():
            print(f"  {i}: {n}")
        print("\nOur classes:", ", ".join(ours.values()))
        return

    our_by_norm = {norm(n): i for i, n in ours.items()}
    their_by_norm = {norm(n): i for i, n in their.items()}
    id_map = {}
    for pair in filter(None, (p.strip() for p in args.map.split(","))):
        t, o = (x.strip() for x in pair.split("="))
        if norm(t) not in their_by_norm:
            raise SystemExit(f"'{t}' is not a class in the downloaded dataset (use --list)")
        if norm(o) not in our_by_norm:
            raise SystemExit(f"'{o}' is not one of our classes: {list(ours.values())}")
        id_map[their_by_norm[norm(t)]] = our_by_norm[norm(o)]
    if not id_map:
        raise SystemExit("No mapping given. Use --map (see --list for class names).")

    split_dst = {"train": "train", "valid": "val", "val": "val", "test": "train"}
    kept = skipped = 0
    for split, out_split in split_dst.items():
        img_dir, lbl_dir = src / split / "images", src / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
            lbl = lbl_dir / (img.stem + ".txt")
            lines_out = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    if cid not in id_map:
                        continue
                    box = to_bbox([float(v) for v in parts[1:]])
                    lines_out.append(f"{id_map[cid]} " + " ".join(f"{v:.6f}" for v in box))
            if not lines_out:
                skipped += 1
                continue
            name = f"{args.prefix}_{out_split}_{img.name}"
            (dst / "images" / out_split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / out_split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / out_split / name)
            (dst / "labels" / out_split / (Path(name).stem + ".txt")).write_text("\n".join(lines_out) + "\n")
            kept += 1

    print(f"Merged {kept} images, skipped {skipped} with no mapped objects.")


if __name__ == "__main__":
    main()
