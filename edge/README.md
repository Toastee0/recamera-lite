# recamera-lite edge (the lite-OS daemon)

The native vision daemon that runs **on** the reCamera. It brings up the camera, runs an
on-device NPU YOLO every frame, decides salience + a threat score, streams H.264 RTSP,
HW-encodes a threat-gated JPEG to tmpfs, and emits UDP salience events to your GPU server.

Target hardware: **reCamera Gen1** (SG2002/CV181x RISC-V, OV5647 sensor) and
**reCamera HQ** (GC2053 sensor), running reCamera-OS (musl, kernel 5.10).

## Files

| File | What |
|------|------|
| `src/vision.cpp`, `src/yolo.{cpp,h}` | the daemon: VI→VPSS (3 channels: RTSP / NPU / JPEG), the NPU YOLO11 decode + NMS, salience/threat, the UDP event sender |
| `src/Makefile` | cross-compile against the Seeed SG2002 SDK |
| `camdetect.sh` | i2c sensor auto-detect → writes `/userdata/sensor.conf` |
| `recamera-lite.conf` | runtime config (server host, cam name, model, thresholds) |
| `S99recamera-lite` | boot autostart; sources both conf files |
| `install.sh` | scp the binary + scripts + model to a device and install the init script |
| `model-build/` | TPU-MLIR recipe: YOLO11n `.pt` → INT8 `.cvimodel` |

## Get the binary

**Prebuilt (no SDK needed):** grab `vision-recamera-sg2002-riscv64-musl` from the
[Releases](https://github.com/Toastee0/recamera-lite/releases) page and skip to *Install*.
It's a stripped RISC-V/musl ELF for the SG2002; it dynamically links the CVITEK MPI + TPU
runtime that already ship **on the device** (`/mnt/system/lib`), so nothing else is needed.

## Build from source

You need the Seeed reCamera **SG2002 SDK** and the **riscv64 musl toolchain** (multi-GB,
not vendored here — get them from <https://github.com/Seeed-Studio/reCamera-OS>). Then:

```bash
export SG200X_SDK_PATH=/path/to/sg2002_recamera_emmc
export RECAM_OS_BUILD=/path/to/reCamera-OS          # SensorSupportList headers
export PATH=/path/to/host-tools/.../bin:$PATH       # riscv64-unknown-linux-musl-g++
cd src && make                                      # produces ./vision
```

## Model

The NPU runs an INT8 `.cvimodel` (YOLO11n COCO-80, 640×640) with a **split detect head**.
Build your own with [`model-build/build_cvimodel.sh`](model-build/build_cvimodel.sh) (runs
in the `sophgo/tpuc_dev` Docker image via TPU-MLIR). The on-device decoder in `yolo.cpp`
expects YOLO11n's 6 conv head outputs — keep that shape if you retrain.

> Note: fused-preprocess is intentionally **off** (a tpu_mlir version segfaults on cv181x),
> so the daemon does the resize (HW letterbox) + INT8 quantize on-device. See the
> model-build README.

## Install onto a device

```bash
./install.sh <device-ip> src/vision model-build/yolo11n_ours.cvimodel
```

Then on the device edit `/userdata/recamera-lite.conf`:

```sh
SERVER=192.168.1.10:23906     # your GPU server
CAM_NAME=auto                 # or a fixed name; the /vis/<name>.jpg URL uses it
MODEL=/userdata/Models/yolo11n_ours.cvimodel
SALCONF=0.35
FPSMAX=15
```

and start it:

```bash
ssh recamera@<device-ip> /etc/init.d/S99recamera-lite restart
```

## Daemon flags (set via the conf, or run by hand)

`vision` is configured by CLI flags (the init script builds them from the conf):

```
-evt host:port     UDP target for salience events (omit = events off)
-name <cam>        camera name (event field + /vis/<cam>.jpg)
-model <path>      .cvimodel path
-salconf <f>       salience confidence threshold (default 0.40)
-fpsmax <f>        threat-FPS cap (default 12; floor is ~2)
-gc2053 | -ov5647  force sensor (else read /userdata/sensor.conf, else OV5647)
-p <port>          RTSP port (default 8554);  -h265  use H.265
-jpegq <n>         JPEG quality (default 80);  -jpegdir <path> (default /run/vis)
-nonpu|-nortsp|-nojpeg   disable a path
-snap | -watch N   one-shot debug helpers
```

Serve the gated JPEG over HTTP by pointing your web server's `/vis` at `/run/vis`
(reCamera-OS ships lighttpd; symlink `/var/www/vis → /run/vis`).
