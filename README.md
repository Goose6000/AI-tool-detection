# Tool detector (YOLO, live camera)

Detects 10 classes: hammer, screwdriver, boxcutter, adjustable_spanner, pliers,
d_clamp, hand_saw, compact_blower, m18_hammer_drill, m18_battery_cp2.
The battery is its own class. A rule decides whether it is seated in the drill.
The detector also logs which tool was REMOVED or RETURNED.

## Workflow

```bash
pip install -r requirements.txt

# 1. Capture photos (SPACE saves, Q quits). Aim for 150+ per class.
python capture.py --label hammer
python capture.py --label drill_with_battery
python capture.py --label drill_no_battery
python capture.py --label battery_loose

# 2. Label the photos with CVAT, Label Studio or labelImg (YOLO format).
#    Keep the class order from data.yaml. Put each image and its .txt in dataset/labeled/.
#    In drill_with_battery photos, draw BOTH a drill box and a battery box.

# 3. Split, train
python split_dataset.py --src dataset/labeled --val 0.2
python train.py --model yolo11s.pt --epochs 100 --device 0

# 4. Run live (headless: add --no-window)
python detect.py --weights runs/detect/tools/weights/best.pt --source 0
```

Output goes to the screen and to `events.jsonl`:
`REMOVED`, `RETURNED`, `BATTERY_IN_TOOL`, `BATTERY_LOOSE`, `BASELINE_RESET`.

## Tuning

- `--min-cover 0.6`: how much of the battery box must lie inside the drill box
  to count as installed. Raise it if loose batteries lying on the drill are
  reported as installed.
- `--window 15 --need 8`: a tool must be seen in 8 of the last 15 frames to
  count as present. This stops a single missed frame from looking like a removal.
- `--conf 0.5`: detection confidence threshold.
- Press `R` in the window to reset the baseline to what is currently visible.

## Training data tips

- Vary backgrounds, lighting, angles and distance. Keep the camera angle you will use in production.
- Take many photos of the M18 drill with and without the battery, from the side and from below.
- If the geometric battery check proves unreliable, the stronger fix is to add a class
  such as `m18_battery_installed` and label installed batteries separately.
  Add it at the END of the list in `data.yaml` and update `logic.py`.
