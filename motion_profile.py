#!/usr/bin/env python3
"""Print a frame-to-frame motion profile to find (a) the static opening to SKIP and
(b) roughly where sustained flight ends. Pure OpenCV, no GPU.
Usage: python motion_profile.py <video> [scene_roi x0 y0 x1 y1]
The ROI should be a patch of the SCENE (not the HUD). Default suits a 720x1600 portrait frame.
"""
import cv2, numpy as np, sys
vid = sys.argv[1]
roi = list(map(int, sys.argv[2:6])) if len(sys.argv) >= 6 else [120, 500, 600, 1100]
cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS)
prev = None; i = 0; step = max(1, int(round(fps * 0.5)))   # sample ~2 Hz
while True:
    ok, f = cap.read()
    if not ok: break
    i += 1
    if i % step: continue
    g = cv2.resize(cv2.cvtColor(f[roi[1]:roi[3], roi[0]:roi[2]], cv2.COLOR_BGR2GRAY), (160, 200))
    if prev is not None:
        m = float(np.mean(np.abs(g.astype(int) - prev)))
        print(f"{i/fps:6.1f}s  {'#'*int(min(m,40)):40}  {m:.1f}")
    prev = g
cap.release()
print("\nLow bars = static/hover (skip these).  Sustained high = active flight.")
