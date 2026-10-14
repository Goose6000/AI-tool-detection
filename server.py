"""Phone-camera detection server, and the backend the ToolLighthouse website reads.

The phone opens the page served here, crops the detection square out of its camera
feed and sends it as a JPEG over a WebSocket. The server runs the model on that
square, replies with the detections, and keeps one station state that the website
polls.

Usage:
  .venv/bin/python server.py --weights runs/detect/tools-4/weights/best.pt
  cloudflared tunnel --url http://localhost:8000     # gives the phone an https URL

Endpoints:
  GET  /                            phone camera page (add ?key=... if --camera-key is set)
  WS   /ws                          frames in, detections out
  GET  /stations/station-01/state   held station state in the website's StationState shape
"""
import argparse
import asyncio
import hmac
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from ultralytics import YOLO

from logic import Stabilizer

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/detect/tools-4/weights/best.pt")
ap.add_argument("--tools", default="tools.yaml")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--window", type=int, default=6, help="frames looked back over")
ap.add_argument("--need", type=int, default=4, help="frames a tool must be seen in to count")
ap.add_argument("--confirm", type=float, default=2, help="seconds a tool must stay detected before the website shows it")
ap.add_argument("--hold", type=float, default=60, help="seconds a result stays shown after the tools are lifted")
ap.add_argument("--camera-timeout", type=float, default=5, help="seconds without frames before the camera counts as offline")
ap.add_argument("--station", default="station-01")
ap.add_argument("--log", default="events.jsonl")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

# Secrets come from the environment so they stay out of shell history and `ps`.
READ_TOKEN = os.environ.get("STATION_TOKEN", "")   # website sends it as a Bearer token
CAMERA_KEY = os.environ.get("CAMERA_KEY", "")      # phone page sends it as ?key=

model = YOLO(args.weights)
model_lock = threading.Lock()
TOOLS = yaml.safe_load(Path(args.tools).read_text())
missing = set(model.names.values()) - set(TOOLS)
if missing:
    raise SystemExit(f"No entry in {args.tools} for: {', '.join(sorted(missing))}")
UUIDS = {name: t["uuid"] for name, t in TOOLS.items()}

STATIC = Path(__file__).parent / "static"
app = FastAPI()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z") if t else None


def log_event(kind, **data):
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **data}
    print(json.dumps(rec))
    with open(args.log, "a") as f:
        f.write(json.dumps(rec) + "\n")


class Station:
    """The one authoritative station state.

    A tool is confirmed once it has been detected for `confirm` seconds in a row
    (the Stabilizer bridges single missed frames). Only confirmed tools reach the
    website, which filters out brief false positives.

    A scan starts when confirmed tools appear in the square. Every tool seen until the square
    is empty again joins that scan. After the square empties, the result is held for
    `hold` seconds, so the student can lift the tools and still follow the drawers.
    Tools appearing again start a new scan that replaces the held one.
    """

    def __init__(self, hold, camera_timeout, confirm):
        self.hold, self.camera_timeout, self.confirm = hold, camera_timeout, confirm
        self.since = {}           # name -> when its current run of detections began
        self.present = set()      # confirmed tools in the square right now
        self.tools = {}           # name -> best confidence, for the current scan
        self.scan_id = None
        self.last_scan_at = None
        self.emptied_at = None
        self.raw_seen = False     # something detected, not yet confirmed
        self.last_frame_at = None

    def update(self, stable, dets):
        """Feed one frame. Returns the confirmed tools."""
        now = time.time()
        self.last_frame_at = now
        seen = {d["name"] for d in dets} | set(stable)
        self.since = {n: self.since.get(n, now) for n in seen}
        present = {n for n in stable if now - self.since[n] >= self.confirm}
        self.raw_seen = bool(seen - present)
        if present and not self.present:
            self.tools, self.scan_id = {}, str(uuid.uuid4())
            log_event("SCAN_STARTED", scan_id=self.scan_id)
        if present:
            self.emptied_at = None
            best = {}
            for d in dets:
                best[d["name"]] = max(best.get(d["name"], 0), d["conf"])
            for n in present:
                if n not in self.tools:
                    self.last_scan_at = now
                self.tools[n] = max(self.tools.get(n, 0), best.get(n, 0))
        elif self.present:
            self.emptied_at = now
        for n in sorted(present - self.present):
            log_event("TOOL_IN_SQUARE", tool=n, uuid=UUIDS[n], scan_id=self.scan_id)
        for n in sorted(self.present - present):
            log_event("TOOL_LEFT_SQUARE", tool=n, uuid=UUIDS[n], scan_id=self.scan_id)
        self.present = present
        return present

    def state(self):
        now = time.time()
        camera_up = self.last_frame_at is not None and now - self.last_frame_at <= self.camera_timeout
        holding = self.tools and (self.present or (self.emptied_at and now - self.emptied_at <= self.hold))
        if not camera_up:
            mode, message = "offline", "La cámara de la estación no está enviando imágenes."
        elif holding:
            mode, message = "held", None
        elif self.raw_seen:
            mode, message = "scanning", None  # detected, not confirmed yet
        else:
            mode, message = "idle", None
        detections = []
        if mode == "held":
            for n in sorted(self.tools):
                website_id = TOOLS[n].get("website_id")
                if website_id:
                    detections.append({"id": UUIDS[n], "toolId": website_id,
                                       "confidence": round(self.tools[n], 3)})
        return {
            "stationId": args.station,
            "mode": mode,
            "source": "live",
            "scanId": self.scan_id if mode == "held" else None,
            "updatedAt": iso(now if not camera_up else self.last_frame_at),
            "lastScanAt": iso(self.last_scan_at),
            "connectivity": {
                "camera": "online" if camera_up else "offline",
                "inference": "online",
                "ledController": "unknown",
            },
            "detections": detections,
            "message": message,
        }


station = Station(args.hold, args.camera_timeout, args.confirm)


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


def same(a, b):
    return hmac.compare_digest(a.encode(), b.encode())


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/stations/{station_id}/state")
def station_state(station_id: str, request: Request):
    if READ_TOKEN and not same(request.headers.get("authorization", ""), f"Bearer {READ_TOKEN}"):
        raise HTTPException(401, "Unauthorized")
    if station_id != args.station:
        raise HTTPException(404, "Station not found")
    return station.state()


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    if CAMERA_KEY and not same(sock.query_params.get("key", ""), CAMERA_KEY):
        await sock.close(code=4401)  # after accept, so the page can read the code
        return
    loop = asyncio.get_running_loop()
    stab = Stabilizer(args.window, args.need)
    try:
        while True:
            jpeg = await sock.receive_bytes()
            t0 = time.perf_counter()
            dets = await loop.run_in_executor(None, detect, jpeg)
            if dets is None:
                await sock.send_json({"error": "bad frame"})
                continue
            present = station.update(stab.update(d["name"] for d in dets), dets)
            await sock.send_json({
                "detections": dets,
                "stable": [{"name": n, "uuid": UUIDS[n]} for n in sorted(present)],
                "ms": round((time.perf_counter() - t0) * 1000),
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
