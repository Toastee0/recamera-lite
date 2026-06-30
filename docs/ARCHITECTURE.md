# Architecture & wire protocol

## Why two tiers

A single always-on heavy detector is wasteful: most frames are boring. recamera-lite
splits the work by cost:

- **Tier 1 — edge NPU (cheap, always on).** The reCamera's SG2002 has a small TPU. It runs
  an INT8 YOLO on every frame at low power and decides *whether anything salient is present*
  (person, animal, vehicle — configurable). This is a gate, not the final answer.
- **Tier 2 — your GPU (expensive, on demand).** A full-size YOLO (e.g. yolo11x at high
  resolution) runs **only on the frames the edge gated**. It confirms, upscales, and gives
  accurate boxes/labels — but it never touches the idle stream.

The edge never streams frames to the GPU continuously. It writes the *one* triggering JPEG
to its own tmpfs and tells the server "now" via a tiny UDP datagram. **The event is the
frame's trigger** — no clock-sync, no continuous pull.

## FPS = threat level

The edge computes a `threat` score per active window from the salient detection:

```
threat = classW · (0.25 + 1.5·area + 10·looming + 0.15·(count-1)) · (0.5 + 0.5·conf)   (clamped 0..1)
```

where `area` = box area / frame area, `looming` = positive area growth vs the previous
frame (something approaching), `count` = number of salient detections, `classW` = 1.0 for a
person else 0.6. The capture/confirm rate scales with it: `fps = fps_min + (fps_max−fps_min)·threat`.
A person walking toward the camera drives high FPS; a distant idle cat, low. Empty scene ⇒ 0.

## The pipeline, end to end

1. Edge NPU detects salience for `START_N` consecutive frames → **START** event.
2. While present, the edge sends an **active** heartbeat every ~2 s and keeps the gated
   JPEG fresh at the threat-derived FPS.
3. The server, on START, begins pulling `http://<edge-ip>/vis/<cam>.jpg` at that FPS
   (capped by `PULL_CAP`) and runs GPU YOLO on each pull.
4. GPU detections in `CONFIRM_CLASSES` are logged as **confirmed sightings**; the annotated
   latest frame is served to the dashboard.
5. After `END_N` empty frames the edge sends **END**; the server stops pulling. A 6 s
   stale-watchdog force-ends a window if events stop arriving.

## Wire protocol

### Events — UDP JSON datagram, default port `23906`

The edge sends to a single configured `host:port` (unicast). Boxes are in **source
1920×1080** coordinates.

`start` / `active`:
```json
{
  "cam": "frontdoor",
  "event": "start",
  "label": "person",
  "conf": 0.82,
  "box": [x1, y1, x2, y2],
  "threat": 0.71,
  "fps": 9.0,
  "dets": [{"l": "person", "c": 0.82, "b": [x1,y1,x2,y2]}, {"l": "dog", "c": 0.55, "b": [...]}],
  "img": "frontdoor.jpg",
  "ts": 1751310000.123,
  "seq": 42
}
```
- `box` / `label` / `conf` — the single top-salient detection (the gate).
- `dets` — the full-scene array (abbreviated keys `l`/`c`/`b`), capped to fit one datagram.
- `active` is an identical ~2 s heartbeat while the subject stays in view.

`end`:
```json
{"cam": "frontdoor", "event": "end", "ts": 1751310012.4, "seq": 77, "dur": 12.3}
```

### Gated frame — HTTP

The edge HW-encodes a 1280×720 JPEG, atomic-writes it to tmpfs, and serves it (lighttpd on
:80) at `http://<edge-ip>/vis/<cam>.jpg`. The server fetches the source IP of the datagram.
Override the URL shape with the server's `VIS_URL` env (`http://{host}/vis/{cam}.jpg`).

## Server API

| Route | Purpose |
|-------|---------|
| `GET /` | dashboard |
| `GET /health` | model/device/port status |
| `GET /api/events` | per-cam state (active, threat, fps, edge + GPU detections) + recent transitions |
| `GET /api/sightings` | recent confirmed sightings |
| `GET /api/latest/<cam>` | the latest annotated frame (JPEG) for an active cam |
| `POST /detect` | run the GPU model on a posted JPEG body (no edge needed) → detections |

## Swapping models

- **Edge**: any YOLO compiled to an INT8 `.cvimodel` with a split detect head — see
  [`../edge/model-build`](../edge/model-build). The decoder in `yolo.cpp` expects YOLO11n's
  6-output split head; retrain/swap weights, keep the head shape.
- **Server**: set `YOLO_MODEL` to any Ultralytics `.pt` (n/s/m/l/x or a custom-trained one).
  Bigger model + higher `YOLO_IMGSZ` = better on small/distant subjects, slower per frame.
