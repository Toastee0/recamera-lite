#!/usr/bin/env python3
# Frame -> 640x640 letterboxed INT8 planar RGB, matching the daemon's NPU input
# (yolo.cpp: pad 114, RGB planar, int8 = round(px/255*qscale), qscale=126.999).
import sys, json, numpy as np
from PIL import Image

src, out_raw, out_meta = sys.argv[1], sys.argv[2], sys.argv[3]
QSCALE = 126.999
img = Image.open(src).convert('RGB')
W, H = img.size
scale = min(640.0/W, 640.0/H)
nw, nh = round(W*scale), round(H*scale)
resized = img.resize((nw, nh), Image.BILINEAR)
canvas = np.full((640, 640, 3), 114, np.uint8)
px, py = (640-nw)//2, (640-nh)//2
canvas[py:py+nh, px:px+nw] = np.asarray(resized)
q = np.round(canvas.astype(np.float32)/255.0*QSCALE).clip(-128, 127).astype(np.int8)
chw = np.ascontiguousarray(q.transpose(2, 0, 1))          # [3,640,640] planar
chw.tofile(out_raw)
json.dump({'W': W, 'H': H, 'scale': scale, 'pad_x': px, 'pad_y': py}, open(out_meta, 'w'))
print(f"preprocessed {src} {W}x{H} -> {out_raw} (scale={scale:.4f} pad=({px},{py}))")
