#!/bin/bash
# Runs INSIDE the sophgo/tpuc_dev container. Workspace mounted at /work, sscma repo at /sscma.
# Produces yolo11n_detection_cv181x_int8.cvimodel (640x640, INT8, COCO-80) for the SG2002 NPU.
set -e
cd /work

# 1) toolchain (pinned-ish; tpuc_dev base is ubuntu22.04/py3.10)
pip install -q ultralytics 2>&1 | tail -1
pip install -q tpu_mlir[onnx] 2>&1 | tail -1
echo "ultralytics: $(python3 -c 'import ultralytics;print(ultralytics.__version__)')"
echo "tpu_mlir: $(python3 -c 'import tpu_mlir;print(getattr(tpu_mlir,"__version__","?"))' 2>/dev/null || pip show tpu_mlir 2>/dev/null | awk '/Version/{print $2}')"

# 2) yolo11n.pt -> ONNX (640, full export; export.py will split the head via --output_names)
if [ ! -f yolo11n.onnx ]; then
  yolo export model=/work/yolo11n.pt format=onnx imgsz=640 opset=14 2>&1 | tail -3
fi
echo "=== onnx detect-head conv output node names (verify they match the recipe) ==="
python3 - <<'PY'
import onnx
m=onnx.load("/work/yolo11n.onnx")
names=[n.name for n in m.graph.node if n.op_type=="Conv" and "cv2" in n.output[0]+n.name or "cv3" in n.output[0]+n.name]
# print conv outputs around the detect head (model.23)
outs=[o for n in m.graph.node for o in n.output if "/model.23/cv" in o and o.endswith("Conv_output_0")]
print("\n".join(sorted(set(outs))))
PY

# 3) ONNX -> cvimodel via the sscma export wrapper (model_transform -> calibration -> deploy)
ON="/model.23/cv2.0/cv2.0.2/Conv_output_0,/model.23/cv3.0/cv3.0.2/Conv_output_0,/model.23/cv2.1/cv2.1.2/Conv_output_0,/model.23/cv3.1/cv3.1.2/Conv_output_0,/model.23/cv2.2/cv2.2.2/Conv_output_0,/model.23/cv3.2/cv3.2.2/Conv_output_0"
python3 /sscma/scripts/export.py /work/yolo11n.onnx \
  --output_names "$ON" \
  --dataset /work/coco128/images/train2017 \
  --test_input /sscma/images/bus.jpg \
  --chip cv181x --quantize INT8 --epoch 128 \
  --work_dir /work/work_dir

echo "=== artifacts ==="
find /work -name '*.cvimodel' -newer /work/yolo11n.pt 2>/dev/null
