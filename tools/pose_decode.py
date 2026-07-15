#!/usr/bin/env python3
# Decode YOLO11-pose 9-tensor NPU output -> person boxes + 17 keypoints, overlay skeleton.
import sys, json, numpy as np
from PIL import Image, ImageDraw

prefix, meta_path, frame, out_img = sys.argv[1:5]
CONF_T = float(sys.argv[5]) if len(sys.argv) > 5 else 0.20
meta = json.load(open(meta_path))

shapes = {}
for line in open(prefix + '.manifest.txt'):
    p = line.split(); shapes[int(p[0])] = list(map(int, p[2:]))
def load(i): return np.fromfile(f'{prefix}.out{i}.bin', dtype=np.float32).reshape(shapes[i])[0]
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

# (box_idx, cls_idx, kpt_idx, stride)
SCALES = [(0, 3, 6, 8), (1, 4, 7, 16), (2, 5, 8, 32)]
proj = np.arange(16, dtype=np.float32)
dets = []
for box_i, cls_i, kpt_i, stride in SCALES:
    box, cls, kpt = load(box_i), load(cls_i), load(kpt_i)   # [64,G,G] [1,G,G] [51,G,G]
    conf = sigmoid(cls[0])
    ys, xs = np.where(conf > CONF_T)
    for gy, gx in zip(ys, xs):
        dfl = box[:, gy, gx].reshape(4, 16)
        e = np.exp(dfl - dfl.max(1, keepdims=True)); sm = e / e.sum(1, keepdims=True)
        d = (sm * proj).sum(1)                              # l,t,r,b (stride units)
        cx, cy = gx + 0.5, gy + 0.5
        x1, y1 = (cx - d[0]) * stride, (cy - d[1]) * stride
        x2, y2 = (cx + d[2]) * stride, (cy + d[3]) * stride
        k = kpt[:, gy, gx].reshape(17, 3)
        kx = (k[:, 0] * 2.0 + gx) * stride
        ky = (k[:, 1] * 2.0 + gy) * stride
        kv = sigmoid(k[:, 2])
        dets.append((x1, y1, x2, y2, float(conf[gy, gx]), np.stack([kx, ky, kv], 1)))

def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1]); xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0
dets.sort(key=lambda d: d[4], reverse=True)
keep = []
for d in dets:
    if all(iou(d, k) < 0.45 for k in keep): keep.append(d)
keep = keep[:5]
print(f"{len(dets)} raw dets > {CONF_T}; {len(keep)} after NMS" + (f"; top conf={keep[0][4]:.3f}" if keep else "; NONE"))

sc, pxpad, pypad = meta['scale'], meta['pad_x'], meta['pad_y']
def unmap(x, y): return (x - pxpad) / sc, (y - pypad) / sc
im = Image.open(frame).convert('RGB'); dr = ImageDraw.Draw(im)
EDGES = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
for (x1, y1, x2, y2, c, kpts) in keep:
    X1, Y1 = unmap(x1, y1); X2, Y2 = unmap(x2, y2)
    dr.rectangle([X1, Y1, X2, Y2], outline=(0, 255, 0), width=3)
    dr.text((X1, max(0, Y1 - 12)), f"person {c:.2f}", fill=(0, 255, 0))
    pts = [(*unmap(kx, ky), kv) for kx, ky, kv in kpts]
    for a, b in EDGES:
        if pts[a][2] > 0.3 and pts[b][2] > 0.3:
            dr.line([pts[a][0], pts[a][1], pts[b][0], pts[b][1]], fill=(255, 80, 0), width=3)
    for KX, KY, kv in pts:
        if kv > 0.3:
            dr.ellipse([KX-5, KY-5, KX+5, KY+5], fill=(0, 200, 255))
im.save(out_img, quality=92)
print("saved", out_img)
