#!/usr/bin/env python3
"""Build a NORTH-UP bird's-eye (BEV) orthographic image of the reconstruction, plus a
car-free version. Uses the compass heading (burned into the HUD) + the VGGT camera poses to
find which way is north in the arbitrary reconstruction frame, then rotates the BEV north-up
(so it matches a north-up satellite map).

Writes: north_bev.png (with cars, for the yellow-line fit & display) and
        north_bev_ground.png (car-free, height-filtered).

!!! HUD-SPECIFIC: the north convention. Read the compass in the video: which IMAGE direction
points north?  For this video N = image-RIGHT, i.e. the camera +x axis -> NORTH=x (default).
Set env NORTH to x | -x | y | -y accordingly for a different video.

Usage: python north_bev.py <points.ply> <cameras.npz> <out_dir>
Env:   RES=1100  BAND=0.022 (ground-slab half-thickness, plane units)  NORTH=x
"""
import sys, os, numpy as np
from PIL import Image
ply, npz, outp = sys.argv[1], sys.argv[2], sys.argv[3]
RES = int(os.environ.get("RES", "1100")); BAND = float(os.environ.get("BAND", "0.022"))
NORTH = os.environ.get("NORTH", "x")

# ---- load + robust-trim point cloud ----
xyz, rgb = [], []
with open(ply) as f:
    line = ""
    while not line.startswith("end_header"): line = f.readline()
    for line in f:
        v = line.split()
        if len(v) >= 6:
            xyz.append((float(v[0]), float(v[1]), float(v[2]))); rgb.append((int(v[3]), int(v[4]), int(v[5])))
xyz = np.asarray(xyz, np.float64); rgb = np.asarray(rgb, np.uint8)
lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0); m = ((xyz >= lo) & (xyz <= hi)).all(1)
xyz, rgb = xyz[m], rgb[m]

# ---- PCA ground basis (2 in-plane axes + normal) ----
c = xyz.mean(0); X = xyz - c
_, _, Vt = np.linalg.svd(X, full_matrices=False)
u_ax, v_ax, n_ax = Vt[0], Vt[1], Vt[2]
uu, vv, hh = X @ u_ax, X @ v_ax, X @ n_ax

# ---- north direction in the reconstruction frame (from poses + compass convention) ----
extr = np.load(npz, allow_pickle=True)["extrinsics"]      # (N,3,4) world->cam
axis = {"x": 0, "-x": 0, "y": 1, "-y": 1}[NORTH]; sign = -1.0 if NORTH.startswith("-") else 1.0
Nu = Nv = 0.0
for R in extr[:, :, :3]:
    nw = sign * R[axis, :]                 # camera axis (image right/up) -> world = that row of R
    ng = nw - (nw @ n_ax) * n_ax           # project onto ground plane
    a, b = ng @ u_ax, ng @ v_ax; r = np.hypot(a, b) or 1; Nu += a / r; Nv += b / r
rot = np.pi / 2 - np.arctan2(Nv, Nu)       # rotate so north -> +v (up)
ca, sa = np.cos(rot), np.sin(rot)
print(f"north rotation = {np.degrees(rot):.1f} deg")

def render(sel, name):
    u, v, col, h = uu[sel], vv[sel], rgb[sel], hh[sel]
    ru, rv = u * ca - v * sa, u * sa + v * ca
    amin, amax, bmin, bmax = ru.min(), ru.max(), rv.min(), rv.max()
    sc = (RES - 20) / max(amax - amin, bmax - bmin, 1e-9)
    ia = np.clip(((ru - amin) * sc + 10).astype(int), 0, RES - 1)
    ib = np.clip((RES - 1 - ((rv - bmin) * sc + 10)).astype(int), 0, RES - 1)
    hsign = 1.0 if ((h - h.mean()) ** 3).mean() >= 0 else -1.0   # draw ground first, cars on top
    img = np.zeros((RES, RES, 3), np.uint8); order = np.argsort(hsign * h)
    img[ib[order], ia[order]] = col[order]
    Image.fromarray(img).save(os.path.join(outp, name)); print("wrote", name)

render(np.ones(len(uu), bool), "north_bev.png")
render(np.abs(hh - np.median(hh)) < BAND, "north_bev_ground.png")   # car-free (thin ground slab)
