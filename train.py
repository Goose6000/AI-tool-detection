"""Train the tool detector.

Usage:  python train.py --model yolo11s.pt --epochs 100
Best weights end up in runs/detect/tools/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained start point (yolo11n/s/m.pt)")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name="tools",
        patience=30,
        # Augmentation tuned for handheld tools on a bench: rotation and flips
        # are safe, strong hue shifts are not (the red M18 branding matters).
        degrees=15,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.1,
    )
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
