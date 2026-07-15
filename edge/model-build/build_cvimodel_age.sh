#!/bin/bash
# Runs INSIDE sophgo/tpuc_dev (mount /work=tpu-work). InsightFace genderage.onnx ->
# INT8 cvimodel for cv181x. Output fc1[1,3] = [female, male, age*100]. Preprocess per
# insightface Attribute: RGB, mean 0, std 1 (raw 0-255 pixels).
# Calibration: REAL face crops (/work/facecal), NOT coco128 — calibrating an age head
# on street scenes would wreck the INT8 ranges.
# Manual 3-step (export.py's fused-preprocess deploy segfaults on cv181x — known).
set -e
pip install -q tpu_mlir[onnx] 2>&1 | tail -1
mkdir -p /work/work_dir_age && cd /work/work_dir_age

model_transform \
  --model_name genderage \
  --model_def /work/genderage.onnx \
  --input_shapes [[1,3,96,96]] \
  --mean 0.0,0.0,0.0 --scale 1.0,1.0,1.0 \
  --pixel_format rgb --channel_format nchw \
  --mlir genderage.mlir 2>&1 | tail -3

run_calibration genderage.mlir \
  --dataset /work/facecal --input_num 128 \
  -o genderage_calib_table 2>&1 | tail -3

model_deploy \
  --mlir genderage.mlir --quantize INT8 --quant_input --processor cv181x \
  --calibration_table genderage_calib_table \
  --model /work/genderage_cv181x_int8.cvimodel 2>&1 | tail -3

ls -la /work/genderage_cv181x_int8.cvimodel
