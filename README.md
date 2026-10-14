# Tool detector (YOLO, live camera)

Detects 10 classes: hammer, screwdriver, boxcutter, adjustable_spanner, pliers,
d_clamp, hand_saw, compact_blower, m18_hammer_drill, m18_battery_cp2.
The battery is its own class. A rule decides whether it is seated in the drill.
The detector also logs which tool was REMOVED or RETURNED.

There are two ways to run it:

- **Phone camera → cloud VM** (`server.py`): a phone streams its camera to the VM,
  and the model reports the UUID of each tool inside the detection square.
- **Local webcam** (`detect.py`): a camera plugged into the same machine, with an OpenCV window.

Current model: `runs/detect/tools-4/weights/best.pt` (yolo11s, 80 epochs, mAP50 0.925).

## Phone camera (cloud VM)

### Setup (once)

```bash
sudo dnf install -y python3.12 python3.12-pip
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip uninstall -y opencv-python && .venv/bin/pip install opencv-python-headless

# Phones only allow the camera over https. A Cloudflare tunnel provides it without opening firewall ports.
sudo dnf install -y https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm
```

The VM has no screen, so it needs the headless OpenCV build. The normal build fails with `libGL.so.1` missing.

### Start

```bash
.venv/bin/python server.py --weights runs/detect/tools-4/weights/best.pt
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` prints an `https://....trycloudflare.com` URL. Open it on the phone and tap
**Start camera**. Put a tool inside the white square. Once the tool is detected steadily,
the page shows its name and UUID.

- The URL changes every time the tunnel restarts. It is public and has no login.
- Both processes stop when the VM reboots or the shell that started them closes.

### Stop

```bash
pkill -f "python server.py"
pkill -f "cloudflared tunnel"
```

### How it works

1. The phone page (`static/index.html`) crops the detection square out of the camera
   image, scales it to 640×640 and sends it as a JPEG over a WebSocket (`/ws`).
2. The server runs the model on the square and replies with the boxes, which the phone draws.
   The phone sends the next frame only after the reply arrives, so no backlog builds up.
   Expect 3–4 fps on the 2-CPU VM. Use `--imgsz 480` for about twice the speed.
3. A tool counts as in the square once it is seen in 4 of the last 6 frames.

### Tool UUIDs

`tools.yaml` maps each class to a fixed UUID. The website should identify tools by these
UUIDs, not by their names. Replace them with the website's own IDs if it already has some.
The server refuses to start if any class has no UUID.

### Output

- `GET /api/latest`: the tools currently in the square, for the website to read:
  `{"t": "2026-10-14T10:02:11", "tools": [{"name": "hammer", "uuid": "3f8a1c52-..."}]}`
- `events.jsonl`: one line each time a tool enters or leaves the square
  (`TOOL_IN_SQUARE`, `TOOL_LEFT_SQUARE`, with name and UUID).

### Options

- `--conf 0.5`: detection confidence threshold.
- `--window 6 --need 4`: frames a tool must be seen in before it counts.
- `--imgsz 640`: model input size.
- `--host 127.0.0.1 --port 8000`: where the server listens. The tunnel connects to it locally.
- To change the size of the square, edit `SQUARE = 0.8` at the top of `static/index.html`.
  It is a fraction of the camera image's shorter side.

## Training and local webcam

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
python detect.py --weights runs/detect/tools-4/weights/best.pt --source 0
```

Output goes to the screen and to `events.jsonl`:
`REMOVED`, `RETURNED`, `BATTERY_IN_TOOL`, `BATTERY_LOOSE`, `BASELINE_RESET`.

## Tuning (detect.py)

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
