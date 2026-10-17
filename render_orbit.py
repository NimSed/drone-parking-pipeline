#!/usr/bin/env python3
"""Rotating 360 orbit GIF of a colored PLY (orthographic). Numpy + PIL only, no GPU.
Z-axis flip, tilt, zoom and point size are env-tunable. Defaults match what we settled on
for this project (ZSIGN=1 was the orientation that read right-side-up; PS=3 fills gaps).

Usage: python render_orbit.py <points.ply> <out_dir>
Env:   RES=700 NFRAMES=56 TILT=72(0=top-down,90=side) ZOOM=1.5 ZSIGN=1(flip up/down)
       PS=3(point size px) DUR=150(ms/frame; larger=slower) MAXPTS=1200000  SAMPLE_ONLY=0
"""
import sys, os, numpy as np
from PIL import Image
ply = sys.argv[1]; outp = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(ply)
RES = int(os.environ.get("RES", "700")); NFRAMES = int(os.environ.get("NFRAMES", "56"))
TILT = float(os.environ.get("TILT", "72")); ZOOM = float(os.environ.get("ZOOM", "1.5"))
ZSIGN = float(os.environ.get("ZSIGN", "1")); PS = int(os.environ.get("PS", "3"))
DUR = int(os.environ.get("DUR", "150")); MAXPTS = int(os.environ.get("MAXPTS", "1200000"))
SAMPLE_ONLY = os.environ.get("SAMPLE_ONLY", "0") == "1"

xyz, rgb = [], []
with open(ply) as f:
    line = ""
    while not line.startswith("end_header"): line = f.readline()
    for line in f:
        v = line.split()
        if len(v) >= 6: xyz.append((float(v[0]), float(v[1]), float(v[2]))); rgb.append((int(v[3]), int(v[4]), int(v[5])))
xyz = np.asarray(xyz, np.float32); rgb = np.asarray(rgb, np.uint8)
lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0); m = ((xyz >= lo) & (xyz <= hi)).all(1)
xyz, rgb = xyz[m], rgb[m]
xyz[:, 2] *= ZSIGN; xyz -= np.median(xyz, 0)
if len(xyz) > MAXPTS:
    idx = np.random.default_rng(0).choice(len(xyz), MAXPTS, replace=False); xyz, rgb = xyz[idx], rgb[idx]

rad = np.percentile(np.linalg.norm(xyz[:, :2], axis=1), 97)
scale = (RES * 0.42 * ZOOM) / rad
rotx = lambda a: np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], np.float32)
rotz = lambda a: np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
tilt = np.radians(TILT); r = PS // 2

def render(theta):
    q = xyz @ (rotx(tilt) @ rotz(theta)).T
    sx = (q[:, 0] * scale + RES / 2).astype(np.int32)
    sy = (RES / 2 - q[:, 1] * scale).astype(np.int32)
    ok = (sx >= 0) & (sx < RES) & (sy >= 0) & (sy < RES)
    sx, sy, col = sx[ok], sy[ok], rgb[ok]; order = np.argsort(q[ok, 2])
    img = np.zeros((RES, RES, 3), np.uint8); so, yo, xo = col[order], sy[order], sx[order]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            img[np.clip(yo + dy, 0, RES - 1), np.clip(xo + dx, 0, RES - 1)] = so
    return Image.fromarray(img)

if SAMPLE_ONLY:
    render(np.radians(45)).save(f"{outp}/orbit_sample.png"); print("wrote orbit_sample.png")
else:
    frames = [render(2 * np.pi * i / NFRAMES) for i in range(NFRAMES)]
    frames[0].save(f"{outp}/orbit.gif", save_all=True, append_images=frames[1:],
                   duration=DUR, loop=0, optimize=True, disposal=2)
    print("wrote orbit.gif", os.path.getsize(f"{outp}/orbit.gif") // 1024, "KB")
