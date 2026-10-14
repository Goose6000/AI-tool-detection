"""Phone-camera detection server.

The phone opens the page served here, crops the detection square out of its camera
feed and sends it as a JPEG over a WebSocket. The server runs the model on that
square and replies with the detections and the UUID of each tool.

Usage:
  .venv/bin/python server.py --weights runs/detect/tools-4/weights/best.pt
  cloudflared tunnel --url http://localhost:8000     # gives the phone an https URL

Endpoints:
  GET  /             phone camera page
  WS   /ws           frames in, detections out
  GET  /api/latest   tools currently in the square (for the website)
"""
import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ultralytics import YOLO

from logic import Stabilizer

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/detect/tools-4/weights/best.pt")
ap.add_argument("--uuids", default="tools.yaml")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--window", type=int, default=6, help="frames looked back over")
ap.add_argument("--need", type=int, default=4, help="frames a tool must be seen in to count")
ap.add_argument("--log", default="events.jsonl")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

model = YOLO(args.weights)
model_lock = threading.Lock()
UUIDS = yaml.safe_load(Path(args.uuids).read_text())
missing = set(model.names.values()) - set(UUIDS)
if missing:
    raise SystemExit(f"No UUID in {args.uuids} for: {', '.join(sorted(missing))}")

STATIC = Path(__file__).parent / "static"
latest = {"t": None, "tools": []}  # last stable result, served at /api/latest

app = FastAPI()


def log_event(kind, **data):
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **data}
    print(json.dumps(rec))
    with open(args.log, "a") as f:
        f.write(json.dumps(rec) + "\n")


def detect(jpeg: bytes):
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    with model_lock:
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    dets = []
    for b in res.boxes:
        name = model.names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        dets.append({
            "name": name,
            "uuid": UUIDS[name],
            "conf": round(float(b.conf), 3),
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],  # 0..1 within the square
        })
    return dets


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/latest")
def api_latest():
    return latest


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    stab = Stabilizer(args.window, args.need)
    present = set()
    try:
        while True:
            jpeg = await sock.receive_bytes()
            t0 = time.perf_counter()
            dets = await loop.run_in_executor(None, detect, jpeg)
            if dets is None:
                await sock.send_json({"error": "bad frame"})
                continue
            now = stab.update(d["name"] for d in dets)
            if now != present:
                for n in sorted(now - present):
                    log_event("TOOL_IN_SQUARE", tool=n, uuid=UUIDS[n])
                for n in sorted(present - now):
                    log_event("TOOL_LEFT_SQUARE", tool=n, uuid=UUIDS[n])
                present = now
                latest["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                latest["tools"] = [{"name": n, "uuid": UUIDS[n]} for n in sorted(present)]
            await sock.send_json({
                "detections": dets,
                "stable": [{"name": n, "uuid": UUIDS[n]} for n in sorted(present)],
                "ms": round((time.perf_counter() - t0) * 1000),
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
