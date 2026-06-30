# recamera-lite

**Two-tier, salience-gated vision for the [Seeed reCamera](https://www.seeedstudio.com/recamera).**

A cheap, always-on **on-device NPU** detector watches the scene. Only when something
salient appears does the camera (a) fire a small UDP event and (b) hand off the
triggering frame to a **local GPU** running a heavier YOLO that confirms and upscales the
detection. When nothing is happening, the edge is silent and the GPU does **zero** work.

```
   reCamera (edge)                              your server (GPU)
 ┌────────────────────┐                       ┌─────────────────────────┐
 │ camera → NPU yolo  │   UDP event :23906    │ event listener          │
 │ (cheap, always on) │ ────────────────────▶ │   ▼ while "active"       │
 │ salience + threat  │                       │ pull gated frame (HTTP) │
 │ writes gated JPEG  │ ◀──── GET /vis/x.jpg ──│   ▼                     │
 │ to tmpfs (lighttpd)│                       │ GPU YOLO confirm        │
 └────────────────────┘                       │   ▼ dashboard + API     │
                                              └─────────────────────────┘
   FPS = threat level: capture rate scales with how interesting the scene is;
   idle ⇒ ~0 fps ⇒ no traffic, no GPU inference.
```

This repo has two halves:

| Dir | What it is |
|-----|------------|
| [`edge/`](edge/) | the **lite OS daemon** for the reCamera — the native `vision` daemon (RTSP + NPU salience + threat-gated JPEG + UDP events), an installer, a config file, and the on-device model-build recipe |
| [`server/`](server/) | the **GPU server** — one self-contained Python service: UDP event listener + gated-frame pull + in-process [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) on your GPU + a live dashboard and JSON API |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and the full wire protocol.

## Quick start — the server (no camera needed to try it)

```bash
cd server
python3 -m venv .venv && . .venv/bin/activate
# install a CUDA torch for your driver first (see requirements.txt), then:
pip install -r requirements.txt
python3 server.py                      # http://localhost:8092/
```

Test the GPU tier directly with any JPEG — no camera required:

```bash
curl --data-binary @photo.jpg -H 'Content-Type: image/jpeg' http://localhost:8092/detect
```

Or simulate a whole edge device (events + a gated frame) with no hardware:

```bash
sudo python3 tools/simulate_edge.py --img photo.jpg --server 127.0.0.1   # then watch the dashboard
```

## Quick start — the edge

The reCamera daemon is C++ and cross-compiled against the Seeed SG2002 SDK (not vendored
here — multi-GB). Build it, then deploy:

```bash
cd edge/src && make                    # needs SG200X_SDK_PATH + the musl toolchain (see src/Makefile)
cd .. && ./install.sh <device-ip> src/vision model-build/yolo11n_ours.cvimodel
# then set SERVER=<your-gpu-box>:23906 in /userdata/recamera-lite.conf on the device
```

Full edge notes (sensors, model build, config, gotchas): [`edge/README.md`](edge/README.md).

## Configuration at a glance

- **Server** (env vars): `YOLO_MODEL` (n/s/m/l/x), `YOLO_IMGSZ`, `YOLO_CONF`, `EVT_PORT`,
  `HTTP_PORT`, `PULL_CAP`, `CONFIRM_CLASSES`, `SAVE_DIR`. See [`server/README.md`](server/README.md).
- **Edge** (`/userdata/recamera-lite.conf`): `SERVER`, `CAM_NAME`, `MODEL`, `SALCONF`,
  `FPSMAX`. See [`edge/README.md`](edge/README.md).

## License

[Apache-2.0](LICENSE). The reCamera edge daemon links the Seeed/Sophgo CVITEK MPI + TPU
runtime SDKs, which carry their own (vendor) licenses — install those from Seeed; they are
not redistributed here.

## Credits

Distilled from a private home-vision build into a clean, general-purpose release. Targets
the reCamera Gen1 (SG2002/CV181x, OV5647) and reCamera HQ (GC2053). YOLO models via
Ultralytics; on-device INT8 compile via [TPU-MLIR](https://github.com/sophgo/tpu-mlir).
