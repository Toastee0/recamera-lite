#!/usr/bin/env python3
# Decode SCRFD 9-tensor output (scores/bboxes/5-kps x strides 8/16/32, 2 anchors/cell)
# from infer_dump bins; overlay + emit the best face crop (112x112 NHWC raw) for
# mobilefacenet. Frame = snap.rgb (640x640 RGB planar, the NPU's own letterboxed view).
import sys, numpy as np
from PIL import Image, ImageDraw

prefix, snap, out_png, out_crop = sys.argv[1:5]
T = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
S = 640
a = np.fromfile(snap, np.uint8)
img = (np.stack([a[:S*S].reshape(S,S), a[S*S:2*S*S].reshape(S,S), a[2*S*S:].reshape(S,S)], -1)
       if a.size == S*S*3 and snap.endswith('.rgb') else a.reshape(S, S, 3))
im = Image.fromarray(img)

# infer_dump out order observed: 0-2 scores(12800/3200/800), 3-5 bbox, 6-8 kps
CFG = [(8, 0, 3, 6), (16, 1, 4, 7), (32, 2, 5, 8)]
def load(i, cols):
    x = np.fromfile(f'{prefix}.out{i}.bin', np.float32)
    return x.reshape(-1, cols)
faces = []
for stride, si, bi, ki in CFG:
    sc = load(si, 1)[:, 0]
    bb = load(bi, 4); kp = load(ki, 10)
    G = S // stride
    cx = np.tile(np.repeat(np.arange(G), 2), G) * 0        # placeholder, build properly:
    grid = np.stack(np.meshgrid(np.arange(G), np.arange(G)), -1).reshape(-1, 2)  # (x,y) per cell? meshgrid order
    # anchor centers: for each cell (row-major y,x), 2 anchors, center=(x*stride, y*stride)
    # SCRFD flat layout verified on-device 2026-07-15: y varies FAST within the flat
    # index (transposed vs the usual row-major-x-fast assumption) — see MULTITASK_NPU.md
    ys, xs = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
    ctr = np.stack([ys.ravel(), xs.ravel()], -1).repeat(2, 0) * stride
    idx = np.where(sc > T)[0]
    for i in idx:
        cx0, cy0 = ctr[i]
        l, t, r, b = bb[i] * stride
        box = (cx0 - l, cy0 - t, cx0 + r, cy0 + b)
        k = kp[i].reshape(5, 2) * stride
        pts = [(cx0 + k[j, 0], cy0 + k[j, 1]) for j in range(5)]
        faces.append((float(sc[i]), box, pts))
faces.sort(key=lambda f: -f[0])
keep = []
def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1]); xb, yb = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, xb-xa) * max(0, yb-ya)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/u if u > 0 else 0
for f in faces:
    if all(iou(f[1], k[1]) < 0.4 for k in keep): keep.append(f)
print(f"faces>{T}: {len(keep)}" + (f"  best={keep[0][0]:.3f}" if keep else "  NONE"))

dr = ImageDraw.Draw(im)
for conf, (x1, y1, x2, y2), pts in keep[:8]:
    dr.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
    dr.text((x1, max(0, y1-11)), f"face {conf:.2f}", fill=(0, 255, 0))
    for px, py in pts: dr.ellipse([px-3, py-3, px+3, py+3], fill=(0, 200, 255))
im.save(out_png)
print("overlay:", out_png)

if keep:
    conf, (x1, y1, x2, y2), _ = keep[0]
    # square-pad the crop a bit (facenet expects a loose aligned face; no alignment v0)
    w, h = x2-x1, y2-y1; c = max(w, h) * 1.25
    mx, my = (x1+x2)/2, (y1+y2)/2
    x1c, y1c = max(0, int(mx-c/2)), max(0, int(my-c/2))
    x2c, y2c = min(S, int(mx+c/2)), min(S, int(my+c/2))
    crop = Image.fromarray(img[y1c:y2c, x1c:x2c]).resize((112, 112), Image.BILINEAR)
    np.asarray(crop, np.uint8).tofile(out_crop)   # NHWC packed RGB uint8
    crop.save(out_crop + '.png')
    print(f"crop: {out_crop} (from {x1c},{y1c},{x2c},{y2c})")
