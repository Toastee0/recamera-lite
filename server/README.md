# recamera-lite server (GPU side)

One self-contained Python service: it listens for the edge's salience events, pulls the
gated frame, runs YOLO on your GPU, and serves a dashboard + JSON API. Standard-library
HTTP/UDP only — the sole third-party dependency is the YOLO runtime
(Ultralytics + torch + pillow). No web framework, no database.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
# GPU: install a CUDA torch matching your driver FIRST, e.g. for CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The first run downloads the YOLO weights (`yolo11n.pt` by default) automatically.

## Run

```bash
python3 server.py                 # dashboard at http://localhost:8092/
```

As a service: edit and install `recamera-server.service` (systemd template).

## Configuration (environment variables)

| Var | Default | Purpose |
|-----|---------|---------|
| `EVT_PORT` | `23906` | UDP port the edge sends salience events to |
| `HTTP_PORT` | `8092` | dashboard + API port |
| `YOLO_MODEL` | `yolo11n.pt` | Ultralytics model — `n/s/m/l/x` or a path to a custom `.pt` |
| `YOLO_IMGSZ` | `640` | inference resolution; higher = better on small/distant subjects |
| `YOLO_CONF` | `0.35` | confidence threshold |
| `YOLO_DEVICE` | `auto` | `auto` → cuda if available, else `cpu`; or force `cuda`/`cpu` |
| `PULL_CAP` | `8` | max gated-frame pulls/confirms per second per camera |
| `STALE_MS` | `6000` | force-end an active window after this much silence |
| `CONFIRM_CLASSES` | people+animals+vehicles | COCO labels that count as a confirmed sighting |
| `VIS_URL` | `http://{host}/vis/{cam}.jpg` | how to fetch the gated frame (`{host}`=event source IP) |
| `SAVE_DIR` | *(off)* | set a path to persist confirmed frames + JSON sidecars |

A capable GPU (≥8 GB) can run `YOLO_MODEL=yolo11x.pt YOLO_IMGSZ=1280` for the best
accuracy on distant subjects, since it only runs on gated frames.

## Test without a camera

```bash
# GPU tier directly:
curl --data-binary @photo.jpg -H 'Content-Type: image/jpeg' http://localhost:8092/detect

# whole pipeline, simulated edge (serves a frame on :80 + sends events):
sudo python3 ../tools/simulate_edge.py --img photo.jpg --server 127.0.0.1
```

## API

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#server-api).
