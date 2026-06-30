#!/usr/bin/env python3
# recamera-lite server — the GPU half of two-tier salience-gated vision.
#
# The reCamera edge daemon runs a cheap on-device NPU detector and, only when
# something salient is in view, (a) emits a UDP JSON event to this server and
# (b) writes the triggering JPEG to its own tmpfs (served over HTTP). This server
# listens for those events and, while a camera is "active", PULLS that gated frame
# and runs a heavier local YOLO on the GPU to confirm/upscale the detection. When
# nothing is salient, the edge is silent and this server does zero inference.
#
#   edge NPU gates  ──UDP event 23906──▶  this server  ──HTTP pull /vis/<cam>.jpg──▶  GPU YOLO
#
# Single process, standard-library HTTP/UDP only; the one third-party dependency is
# the YOLO model runtime (ultralytics + torch + pillow). No framework, no database.
#
# Run:  python3 server.py        (then open http://localhost:8092/)
# Env:  EVT_PORT(23906) HTTP_PORT(8092) YOLO_MODEL(yolo11n.pt) YOLO_CONF(0.35)
#       YOLO_IMGSZ(640) YOLO_DEVICE(auto) PULL_CAP(8) CONFIRM_CLASSES(comma list)
#       STALE_MS(6000) SAVE_DIR(unset=off)

import io
import json
import os
import socket
import threading
import time
import urllib.request
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

EVT_PORT  = int(os.environ.get("EVT_PORT", "23906"))     # UDP — edge salience events land here
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8092"))     # dashboard + JSON API
MODEL     = os.environ.get("YOLO_MODEL", "yolo11n.pt")   # n/s/m/l/x — x = best accuracy on a big GPU
CONF      = float(os.environ.get("YOLO_CONF", "0.35"))
IMGSZ     = int(os.environ.get("YOLO_IMGSZ", "640"))     # higher = better on small/distant subjects
PULL_CAP  = float(os.environ.get("PULL_CAP", "8"))       # max gated-frame pulls/confirms per sec per cam
STALE_MS  = int(os.environ.get("STALE_MS", "6000"))      # force-end a window after this silence
SAVE_DIR  = os.environ.get("SAVE_DIR", "")               # set a path to persist confirmed frames
# How to fetch the gated frame from the edge device. {host}=event source IP, {cam}=cam name.
# Default matches the edge daemon's lighttpd on :80; override if your frame server differs.
VIS_URL   = os.environ.get("VIS_URL", "http://{host}/vis/{cam}.jpg")
_dev_env  = os.environ.get("YOLO_DEVICE", "auto")
DEVICE    = ("cuda" if torch.cuda.is_available() else "cpu") if _dev_env == "auto" else _dev_env
# Classes that count as a "confirmed sighting" (COCO names). Default: people + animals + vehicles.
CONFIRM   = set(filter(None, os.environ.get(
    "CONFIRM_CLASSES",
    "person,cat,dog,bird,horse,sheep,cow,bear,car,truck,bus,bicycle,motorcycle").split(",")))

# ───────────────────────── model ─────────────────────────
print(f"[yolo] loading {MODEL} on {DEVICE} (imgsz={IMGSZ}) ...", flush=True)
_model = YOLO(MODEL)
_model.predict(Image.new("RGB", (640, 480)), imgsz=IMGSZ, device=DEVICE, verbose=False)  # warm CUDA
_infer_lock = threading.Lock()
print(f"[yolo] ready ({DEVICE}, {MODEL}, imgsz={IMGSZ}, conf>={CONF})", flush=True)


def detect(jpeg_bytes):
    """Run the GPU model on a JPEG. Returns (detections, meta). Serialized — one GPU op
    at a time. detections = [{label, conf, box:[x1,y1,x2,y2]}] in the JPEG's own pixels."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    t0 = time.time()
    with _infer_lock:
        r = _model.predict(img, conf=CONF, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
    ms = int((time.time() - t0) * 1000)
    names = r.names
    dets = []
    for b in r.boxes:
        dets.append({"label": names[int(b.cls[0])],
                     "conf": round(float(b.conf[0]), 3),
                     "box": [round(float(v)) for v in b.xyxy[0].tolist()]})
    return dets, {"model": MODEL, "device": DEVICE, "ms": ms, "size": img.size}


# ───────────────────── per-camera state ─────────────────────
class Cam:
    __slots__ = ("name", "active", "label", "conf", "box", "threat", "fps", "dets",
                 "since", "last_seen", "host", "last_pull", "busy",
                 "latest_jpeg", "latest", "total_starts", "events")

    def __init__(self, name):
        self.name = name
        self.active = False
        self.label = None; self.conf = 0.0; self.box = None
        self.threat = 0.0; self.fps = 0.0; self.dets = []
        self.since = 0; self.last_seen = 0; self.host = None
        self.last_pull = 0.0; self.busy = False
        self.latest_jpeg = None; self.latest = None
        self.total_starts = 0; self.events = 0


cams = {}
cams_lock = threading.Lock()
recent = deque(maxlen=50)       # transition log for the UI
sightings = deque(maxlen=200)   # confirmed person/animal events


def get_cam(name):
    with cams_lock:
        c = cams.get(name)
        if not c:
            c = Cam(name); cams[name] = c
        return c


def now_ms():
    return int(time.time() * 1000)


# ───────────────────── UDP event listener ─────────────────────
def on_event(d, src_ip):
    """Handle one edge datagram. Schema (start/active):
      {cam,event,label,conf,box:[x1,y1,x2,y2],threat,fps,dets:[{l,c,b}],img,ts,seq}
    end: {cam,event:end,ts,seq,dur}. box/dets are in source 1920x1080 coords."""
    name = str(d.get("cam", ""))[:64]
    ev = d.get("event")
    if not name or ev not in ("start", "active", "end"):
        return
    c = get_cam(name)
    c.last_seen = now_ms(); c.events += 1; c.host = src_ip

    if ev in ("start", "active"):
        if d.get("label"): c.label = str(d["label"])[:32]
        if isinstance(d.get("conf"), (int, float)): c.conf = float(d["conf"])
        if isinstance(d.get("threat"), (int, float)): c.threat = float(d["threat"])
        if isinstance(d.get("fps"), (int, float)): c.fps = float(d["fps"])
        if isinstance(d.get("box"), list) and len(d["box"]) == 4: c.box = [float(x) for x in d["box"]]
        if isinstance(d.get("dets"), list): c.dets = d["dets"][:32]
        if not c.active:
            c.active = True; c.since = now_ms(); c.total_starts += 1
            rec = {"type": "start", "cam": name, "ts": c.since, "label": c.label, "conf": c.conf}
            recent.append(rec)
            print(f"[evt] START {name} {c.label} {c.conf:.2f} threat={c.threat:.2f} fps={c.fps:.1f}", flush=True)
    elif ev == "end":
        if c.active:
            c.active = False
            dur = float(d.get("dur", (now_ms() - c.since) / 1000))
            recent.append({"type": "end", "cam": name, "ts": now_ms(), "dur": dur})
            print(f"[evt] END   {name} dur={dur:.1f}s", flush=True)


def udp_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", EVT_PORT))
    print(f"[evt] salience-event listener on udp/{EVT_PORT}", flush=True)
    while True:
        try:
            data, (ip, _port) = s.recvfrom(65535)
            on_event(json.loads(data.decode("utf-8")), ip)
        except (ValueError, UnicodeDecodeError):
            continue
        except Exception as e:
            print("[evt] error:", e, flush=True)


def sweep_loop():
    """Force-end windows that went silent (lost the end + heartbeats)."""
    while True:
        time.sleep(1)
        t = now_ms()
        for c in list(cams.values()):
            if c.active and t - c.last_seen > STALE_MS:
                c.active = False
                recent.append({"type": "end", "cam": c.name, "ts": t,
                               "dur": (t - c.since) / 1000, "stale": True})
                print(f"[evt] END   {c.name} (stale watchdog)", flush=True)


# ───────────────── gated-frame pull + GPU confirm ─────────────────
def annotate(jpeg, edge_dets, gpu_dets, edge_src=(1920, 1080)):
    """Draw edge boxes (dashed yellow, source coords) + GPU boxes (solid, frame coords)
    onto the JPEG so the dashboard can stay dumb. Returns annotated JPEG bytes."""
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception:
        return jpeg
    W, H = img.size
    dr = ImageDraw.Draw(img)
    ex, ey = W / edge_src[0], H / edge_src[1]
    for d in edge_dets or []:
        b = d.get("b") or d.get("box")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = b[0] * ex, b[1] * ey, b[2] * ex, b[3] * ey
        for off in range(0, int(max(x2 - x1, y2 - y1)), 12):  # cheap dashed rect
            dr.line([x1 + off, y1, min(x1 + off + 6, x2), y1], fill=(240, 192, 64), width=2)
            dr.line([x1 + off, y2, min(x1 + off + 6, x2), y2], fill=(240, 192, 64), width=2)
        for off in range(0, int(y2 - y1), 12):
            dr.line([x1, y1 + off, x1, min(y1 + off + 6, y2)], fill=(240, 192, 64), width=2)
            dr.line([x2, y1 + off, x2, min(y1 + off + 6, y2)], fill=(240, 192, 64), width=2)
    for d in gpu_dets or []:
        b = d["box"]
        col = (84, 209, 138) if d["label"] == "person" else (90, 176, 240)
        dr.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=3)
        dr.text((b[0] + 3, max(0, b[1] - 12)), f"{d['label']} {int(d['conf']*100)}%", fill=col)
    out = io.BytesIO(); img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def pull_and_confirm(c):
    """Pull one gated frame from the edge device and run the GPU model on it."""
    if c.busy or not c.host:
        return
    c.busy = True
    ts = now_ms()
    try:
        url = VIS_URL.format(host=c.host, cam=c.name)
        with urllib.request.urlopen(url, timeout=2.5) as r:
            jpeg = r.read()
        dets, meta = detect(jpeg)
        confirmed = [d for d in dets if d["label"] in CONFIRM]
        c.latest_jpeg = annotate(jpeg, c.dets, dets)
        c.latest = {"ts": ts, "edge": {"label": c.label, "conf": c.conf, "box": c.box},
                    "gpu": {"ms": meta["ms"], "model": meta["model"], "detections": dets},
                    "confirmed": [f"{d['label']} {d['conf']:.2f}" for d in confirmed]}
        if confirmed:
            s = {"ts": ts, "cam": c.name, "threat": c.threat,
                 "edge": {"label": c.label, "conf": c.conf},
                 "gpu": [{"label": d["label"], "conf": d["conf"], "box": d["box"]} for d in confirmed]}
            sightings.append(s)
            if SAVE_DIR:
                save_frame(c.name, ts, jpeg, s)
    except Exception:
        pass  # device may have ended the window / be unreachable — ignore, try again next tick
    finally:
        c.busy = False


def save_frame(name, ts, jpeg, meta):
    try:
        day = time.strftime("%Y-%m-%d", time.localtime(ts / 1000))
        d = os.path.join(SAVE_DIR, name, day)
        os.makedirs(d, exist_ok=True)
        base = os.path.join(d, str(ts))
        with open(base + ".jpg", "wb") as f:
            f.write(jpeg)
        with open(base + ".json", "w") as f:
            json.dump(meta, f)
    except Exception as e:
        print("[save] error:", e, flush=True)


def confirm_loop():
    while True:
        time.sleep(0.2)
        for c in list(cams.values()):
            if not c.active:
                continue
            rate = min(c.fps or 3, PULL_CAP)
            if rate <= 0:
                continue
            t = time.time()
            if t - c.last_pull >= 1.0 / rate:
                c.last_pull = t
                threading.Thread(target=pull_and_confirm, args=(c,), daemon=True).start()


# ───────────────────────── HTTP API ─────────────────────────
def snapshot():
    out = {}
    for name, c in list(cams.items()):
        out[name] = {"active": c.active, "label": c.label, "conf": c.conf, "box": c.box,
                     "dets": c.dets if c.active else [], "threat": c.threat, "fps": c.fps,
                     "activeFor": round((now_ms() - c.since) / 1000) if c.active else 0,
                     "host": c.host, "totalStarts": c.total_starts, "events": c.events,
                     "gpu": (c.latest or {}).get("gpu"), "confirmed": (c.latest or {}).get("confirmed", [])}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "public", "index.html"), "rb") as f:
                    return self._bytes(f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._json({"error": "dashboard missing"}, 404)
        if p == "/health":
            return self._json({"ok": True, "model": MODEL, "device": DEVICE,
                               "imgsz": IMGSZ, "evt_port": EVT_PORT, "cams": list(cams)})
        if p == "/api/events":
            return self._json({"cams": snapshot(), "recent": list(recent)[-25:]})
        if p == "/api/sightings":
            return self._json({"recent": list(sightings)[-50:]})
        if p.startswith("/api/latest/"):
            c = cams.get(p[len("/api/latest/"):])
            if not c or not c.latest_jpeg:
                return self._json({"error": "no frame yet"}, 404)
            return self._bytes(c.latest_jpeg, "image/jpeg")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        # POST a JPEG body here to test the GPU tier directly (no edge device needed).
        if self.path.rstrip("/") != "/detect":
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            dets, meta = detect(raw)
        except Exception as e:
            return self._json({"error": f"bad image: {e}"}, 400)
        self._json({"detections": dets, "counts": dict(Counter(d["label"] for d in dets)), **meta})


def main():
    threading.Thread(target=udp_loop, daemon=True).start()
    threading.Thread(target=sweep_loop, daemon=True).start()
    threading.Thread(target=confirm_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(f"[http] dashboard + API on http://0.0.0.0:{HTTP_PORT}/", flush=True)
    if SAVE_DIR:
        print(f"[save] persisting confirmed frames -> {SAVE_DIR}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
