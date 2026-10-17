#!/usr/bin/env python3
"""Rough BEV->satellite registration by aligning the one distinctive feature both share:
the thick YELLOW line (BEV) and the orange WALKWAY (satellite). Fits a line to each and
solves a similarity (rotation+scale+translation). Saves M_yellow.npy (2x3, north_bev->sat).

This is the pragmatic/non-final registration. Proper version would add a 2nd anchor (hatch),
metric scale from altitude, and an ECC refine. Generic feature matchers (SIFT/LoFTR) FAIL here
because the lot is repetitive + cross-domain (cars moved, appearance differs).

!!! SCENE-SPECIFIC: the satellite WALKWAY roi, and that the shared feature is yellow/orange.
Usage: python align_yellow.py <north_bev.png> <satellite.png> <out_dir>
Env:   SAT_ROI="555 350 690 720"   # box around the walkway in the satellite (x0 y0 x1 y1)
"""
import sys, os, cv2, numpy as np
bev_p, sat_p, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
SAT_ROI = tuple(map(int, os.environ.get("SAT_ROI", "555 350 690 720").split()))
bev = cv2.imread(bev_p); sat = cv2.imread(sat_p)

def ymask(img, roi=None):                 # yellow/orange mask
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (12, 70, 70), (40, 255, 255))
    if roi:
        r = np.zeros_like(m); x0, y0, x1, y1 = roi; r[y0:y1, x0:x1] = 255; m = cv2.bitwise_and(m, r)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def fitline(mask):                        # principal direction via 2x2 covariance (fast)
    ys, xs = np.where(mask > 0); P = np.c_[xs, ys].astype(float); cc = P.mean(0); Xp = P - cc
    w, V = np.linalg.eigh(Xp.T @ Xp); dd = V[:, np.argmax(w)]; pr = Xp @ dd
    return cc, dd, pr.min(), pr.max()

cb, db, b0, b1 = fitline(ymask(bev))
cs, ds, s0, s1 = fitline(ymask(sat, SAT_ROI))
if db @ np.array([0, 1]) < 0: db = -db; b0, b1 = -b1, -b0     # make both point the same way
if ds @ np.array([0, 1]) < 0: ds = -ds; s0, s1 = -s1, -s0
ang = np.arctan2(ds[1], ds[0]) - np.arctan2(db[1], db[0])
scl = (s1 - s0) / (b1 - b0)
cA, sA = np.cos(ang) * scl, np.sin(ang) * scl
R = np.array([[cA, -sA], [sA, cA]]); t = cs - R @ cb
M = np.hstack([R, t.reshape(2, 1)]).astype(np.float32)
print(f"rotation {np.degrees(ang):.1f} deg, scale {scl:.3f}")
np.save(os.path.join(outdir, "M_yellow.npy"), M)

warp = cv2.warpAffine(bev, M, (sat.shape[1], sat.shape[0]))
blend = cv2.addWeighted(sat, 0.55, warp, 0.75, 0)
h, w = bev.shape[:2]
proj = cv2.transform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), M)
cv2.polylines(blend, [np.int32(proj)], True, (0, 0, 255), 3)
cv2.imwrite(os.path.join(outdir, "aligned_overlay.jpg"), blend)
print("wrote M_yellow.npy + aligned_overlay.jpg")
