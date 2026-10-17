#!/usr/bin/env python3
"""Quick sanity render: 3 orthographic projections of the PLY (xy / xz / yz).
A flat parking lot should look like a recognizable top-down in ONE of them and near-flat
sheets in the other two. Numpy + PIL only.
Usage: python render_views.py <points.ply> <out_dir>   (env RES=900)
"""
import sys, os, numpy as np
from PIL import Image
ply = sys.argv[1]; outp = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(ply); RES = int(os.environ.get("RES", "900"))
xyz, rgb = [], []
with open(ply) as f:
    line = ""
    while not line.startswith("end_header"): line = f.readline()
    for line in f:
        v = line.split()
        if len(v) >= 6: xyz.append((float(v[0]), float(v[1]), float(v[2]))); rgb.append((int(v[3]), int(v[4]), int(v[5])))
xyz = np.asarray(xyz); rgb = np.asarray(rgb, np.uint8)
lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0); m = ((xyz >= lo) & (xyz <= hi)).all(1); xyz, rgb = xyz[m], rgb[m]
for name, (a, b, dep) in {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}.items():
    pa, pb, pd = xyz[:, a], xyz[:, b], xyz[:, dep]
    sc = (RES - 20) / max(np.ptp(pa), np.ptp(pb), 1e-6)
    ia = np.clip(((pa - pa.min()) * sc + 10).astype(int), 0, RES - 1)
    ib = np.clip((RES - 1 - ((pb - pb.min()) * sc + 10)).astype(int), 0, RES - 1)
    img = np.zeros((RES, RES, 3), np.uint8); order = np.argsort(-pd)
    img[ib[order], ia[order]] = rgb[order]
    Image.fromarray(img).save(f"{outp}/view_{name}.png"); print("wrote", f"view_{name}.png")
