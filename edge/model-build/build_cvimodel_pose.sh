#!/bin/bash
# Runs INSIDE the sophgo/tpuc_dev container (mounts: /work=tpu-work, /sscma=sscma repo).
# Produces yolo11n_pose_cv181x_int8.cvimodel (640x640, INT8) for the SG2002 NPU.
# Pose head = detection head (cv2 box-DFL + cv3 cls) PLUS cv4 keypoints -> 9 output convs
# (vs 6 for detection). Decode is on-device; here we just terminate the graph at the 9
# head convs, calibrate INT8, and deploy — same flow as build_cvimodel.sh.
set -e
cd /work

pip install -q ultralytics 2>&1 | tail -1
pip install -q tpu_mlir[onnx] 2>&1 | tail -1
echo "ultralytics: $(python3 -c 'import ultralytics;print(ultralytics.__version__)')"

# 1) yolo11n-pose.pt -> ONNX (auto-downloads the weights on first use)
if [ ! -f yolo11n-pose.onnx ]; then
  yolo export model=/work/yolo11n-pose.pt format=onnx imgsz=640 opset=14 2>&1 | tail -3
fi

# 2) Discover the 9 head terminal-conv outputs (cv2=box, cv3=cls, cv4=kpt), 3 scales each.
echo "=== model.23 head terminal conv outputs (must be 9: cv2/cv3/cv4 x3) ==="
python3 - <<'PY'
import onnx, re
m = onnx.load("/work/yolo11n-pose.onnx")
outs = sorted({o for n in m.graph.node for o in n.output
               if re.match(r"/model\.23/cv[234]\.\d+/cv[234]\.\d+\.2/Conv_output_0$", o)})
for o in outs: print(o)
print("COUNT", len(outs))
open("/work/pose_output_names.txt","w").write(",".join(outs))
PY
ON="$(cat /work/pose_output_names.txt)"
echo "output_names: $ON"

# 3) ONNX -> cvimodel (INT8, cv181x) via the sscma export wrapper
python3 /sscma/scripts/export.py /work/yolo11n-pose.onnx \
  --output_names "$ON" \
  --dataset /work/coco128/images/train2017 \
  --test_input /sscma/images/bus.jpg \
  --chip cv181x --quantize INT8 --epoch 128 \
  --work_dir /work/work_dir_pose

echo "=== artifacts ==="
find /work -name '*pose*.cvimodel' 2>/dev/null
# normalize the output name
f="$(find /work/work_dir_pose -name '*.cvimodel' 2>/dev/null | head -1)"
[ -n "$f" ] && cp -v "$f" /work/yolo11n_pose_cv181x_int8.cvimodel || echo "NO cvimodel produced"
