# npu-tools — bench & inspect any cvimodel on the reCamera NPU

Small standalone tools for experimenting with models on the SG2002 (CV181x) NPU.
Cross-compile with the same toolchain/flags as the daemon (see `edge/README.md`):

```sh
TPU=<sdk>/install/soc_sg2002_recamera_emmc/tpu_musl_riscv64/cvitek_tpu_sdk
riscv64-unknown-linux-musl-g++ -O2 -std=c++17 -DARCH_CV181X -I$TPU/include \
  -o npu_fwbench npu_fwbench.cpp -L$TPU/lib -lcviruntime -lcvikernel -lcvimath \
  -lpthread -latomic -lm
```

Run on-device with `LD_LIBRARY_PATH=/mnt/system/lib` (runtime libs live there).

- **`npu_fwbench.cpp`** — model-agnostic forward-pass timer. Loads any `.cvimodel`,
  fills the input from a raw file (content irrelevant for timing), runs N forwards,
  reports ms/frame + FPS. Measured on a shed cam: yolo11n detect 33.4 ms (30 fps),
  yolo11n-pose 36.2 ms (27.6 fps) — pose costs ~8% over detect.
- **`npu_posedump.cpp`** — single forward + dump of every output tensor to
  `<prefix>.out<i>.bin` + a manifest, for **INT8-planar (non-fused)** models like the
  ones `model-build/` produces. (The fused-UINT8 pre-builts use `infer_dump` instead.)

Host-side decode for pose lives in `tools/pose_preprocess.py` (frame → letterboxed
INT8 planar) + `tools/pose_decode.py` (9-tensor DFL+keypoint decode, NMS, skeleton
overlay). Compile recipe for pose: `model-build/build_cvimodel_pose.sh` — note the
two-step deploy in its comments (`export.py`'s fused-preprocess deploy segfaults on
cv181x; re-run `model_deploy` without `--fuse_preprocess` from the work dir).
