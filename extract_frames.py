#!/usr/bin/env python3
"""Sample NF frames evenly over [t0,t1], center-square-crop them, and black out the HUD.
Saves f_000.jpg ... and times.npy (each frame's real video timestamp, for later sync).

!!! HUD-SPECIFIC !!!  CROP and MASK below are for THIS DJI-style 720x1600 portrait HUD.
For a new video, open one frame, find the burned-in overlay, and edit CROP/MASK:
  - CROP: a SQUARE region of clean scene (square avoids VGGT's 518x518 aspect squish).
  - MASK: rectangles over any overlay still inside the crop (telemetry, buttons, icons).
Usage: python extract_frames.py <video> <out_dir> <t0_sec> <t1_sec> <num_frames>
"""
import cv2, numpy as np, sys, os
vid, outdir, t0, t1, NF = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
os.makedirs(outdir, exist_ok=True)

CROP = (0, 440, 720, 1160)                       # x0,y0,x1,y1  -> 720x720 center square
MASK = [(0, 440, 130, 730), (630, 440, 720, 525)]  # HUD slivers left telemetry / right button

cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS)
times = np.linspace(t0, t1, NF)
for i, t in enumerate(times):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    if not ok:
        print("miss", i); continue
    for (x0, y0, x1, y1) in MASK:
        f[y0:y1, x0:x1] = 0
    sq = f[CROP[1]:CROP[3], CROP[0]:CROP[2]]
    cv2.imwrite(os.path.join(outdir, f"f_{i:03d}.jpg"), sq)
np.save(os.path.join(outdir, "times.npy"), times)
cap.release()
print(f"extracted {NF} frames  t={times[0]:.1f}..{times[-1]:.1f}s  step={times[1]-times[0]:.3f}s  size={CROP[2]-CROP[0]}x{CROP[3]-CROP[1]}")
