#!/usr/bin/env python3
"""Final deliverable video: 3 panels.
  TOP          = original drone video (rotated to landscape), natural speed, trimmed to segment
  BOTTOM-LEFT  = camera-coverage on the north-up BEV   (persistent/accumulating)
  BOTTOM-RIGHT = camera-coverage on the satellite map  (persistent/accumulating)
Coverage = ground footprint of each frame's view, projected via the poses (BEV) and M (sat).
Sync uses the real per-frame timestamps in times.npy (NOT a uniform assumption).

Usage: python footprint_3panel.py <points.ply> <cameras.npz> <times.npy> <north_bev.png>
                                   <satellite.png> <video> <out.mp4>
Env:   NORTH=x (match north_bev.py)  TW=720  FPS=10  VIDEO_ROT=ccw|cw|none
"""
import sys, os, numpy as np, cv2
ply, npz, times_f, nbimg, sat_p, vid, out = sys.argv[1:8]
D = os.path.dirname(out); RES_NB = 1100; FPS = int(os.environ.get("FPS", "10")); TW = int(os.environ.get("TW", "720"))
NORTH = os.environ.get("NORTH", "x"); VROT = os.environ.get("VIDEO_ROT", "ccw")
COV = (80, 220, 80); CUR = (255, 255, 255)
BH = 430
ROT = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "none": None}[VROT]

# ---- PCA + north rot + pixel mapping (identical to north_bev.py) ----
xyz = []
with open(ply) as f:
    line = ""
    while not line.startswith("end_header"): line = f.readline()
    for line in f:
        v = line.split()
        if len(v) >= 6: xyz.append((float(v[0]), float(v[1]), float(v[2])))
xyz = np.asarray(xyz); lo, hi = np.percentile(xyz, 1, 0), np.percentile(xyz, 99, 0); xyz = xyz[((xyz >= lo) & (xyz <= hi)).all(1)]
c = xyz.mean(0); X = xyz - c
_, _, Vt = np.linalg.svd(X, full_matrices=False); u_ax, v_ax, n_ax = Vt
uu, vv = X @ u_ax, X @ v_ax
d = np.load(npz, allow_pickle=True); extr, intr = d["extrinsics"], d["intrinsics"]
axis = {"x": 0, "-x": 0, "y": 1, "-y": 1}[NORTH]; sign = -1.0 if NORTH.startswith("-") else 1.0
Nu = Nv = 0.0
for R in extr[:, :, :3]:
    nw = sign * R[axis, :]; ng = nw - (nw @ n_ax) * n_ax; a, b = ng @ u_ax, ng @ v_ax; r = np.hypot(a, b) or 1; Nu += a / r; Nv += b / r
rot = np.pi / 2 - np.arctan2(Nv, Nu); ca, sa = np.cos(rot), np.sin(rot)
ru, rv = uu * ca - vv * sa, uu * sa + vv * ca; amin, bmin = ru.min(), rv.min(); sc = (RES_NB - 20) / max(ru.max() - ru.min(), rv.max() - rv.min())
def to_nb(u, v):
    pu = np.asarray(u) * ca - np.asarray(v) * sa; pv = np.asarray(u) * sa + np.asarray(v) * ca
    return np.stack([(pu - amin) * sc + 10, RES_NB - 1 - ((pv - bmin) * sc + 10)], -1)

# ---- footprints (ray-cast image corners onto ground plane), clamp horizon blow-ups ----
W = H = 518; corners = [(0, 0), (W, 0), (W, H), (0, H)]; MAXR = 0.55 * max(np.ptp(uu), np.ptp(vv))
foot_nb = []
for i in range(len(extr)):
    R, t = extr[i, :, :3], extr[i, :, 3]; Rt = R.T; C = -Rt @ t; Ki = np.linalg.inv(intr[i]); nad = ((C - c) @ u_ax, (C - c) @ v_ax); q = []
    for (px, py) in corners:
        dw = Rt @ (Ki @ np.array([px, py, 1.0])); den = dw @ n_ax
        P = C + (((c - C) @ n_ax) / den) * dw if abs(den) > 1e-9 else None
        uv = np.array([(P - c) @ u_ax, (P - c) @ v_ax]) if P is not None else np.array(nad)
        if (not np.isfinite(np.hypot(*(uv - np.array(nad))))) or np.hypot(*(uv - np.array(nad))) > MAXR:
            dh = dw - (dw @ n_ax) * n_ax; d2 = np.array([dh @ u_ax, dh @ v_ax]); nn = np.hypot(*d2) or 1; uv = np.array(nad) + (d2 / nn) * MAXR
        q.append(uv)
    foot_nb.append(to_nb(np.array(q)[:, 0], np.array(q)[:, 1]))
foot_nb = np.array(foot_nb)
M = np.load(D + "/M_yellow.npy"); foot_sat = np.array([(M[:, :2] @ q.T).T + M[:, 2] for q in foot_nb])
times = np.load(times_f); NF = len(times); T0, T1 = times[0], times[-1]

def interp(foots, t):
    i = int(np.clip(np.searchsorted(times, t), 1, NF - 1)); fr = np.clip((t - times[i - 1]) / (times[i] - times[i - 1]), 0, 1)
    return (1 - fr) * foots[i - 1] + fr * foots[i]

nb_base = cv2.imread(nbimg); sat_base = cv2.imread(sat_p)
def cov(base, foots, t):
    cur = int(np.clip(np.searchsorted(times, t), 0, NF - 1)); p = base.copy(); ov = p.copy()
    for j in range(cur + 1): cv2.fillPoly(ov, [np.int32(foots[j])], COV)          # persistent accumulation
    p = cv2.addWeighted(p, 0.55, ov, 0.45, 0)
    cv2.polylines(p, [np.int32(interp(foots, t))], True, CUR, 2, cv2.LINE_AA)       # current footprint
    return p
def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0]); nw, nh = int(img.shape[1] * s), int(img.shape[0] * s)
    o = np.zeros((h, w, 3), np.uint8); o[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = cv2.resize(img, (nw, nh)); return o

cap = cv2.VideoCapture(vid); fps = cap.get(cv2.CAP_PROP_FPS)
_, f0 = cap.read(); f0 = cv2.rotate(f0, ROT) if ROT is not None else f0
toph = int(f0.shape[0] * TW / f0.shape[1]); pw = TW // 2
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (TW, toph + BH))
def label(img, txt, x, y): cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

t = T0
while t <= T1 + 1e-6:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = cap.read()
    if not ok: break
    top = cv2.resize(cv2.rotate(f, ROT) if ROT is not None else f, (TW, toph))
    bl = fit(cov(nb_base, foot_nb, t), pw, BH); br = fit(cov(sat_base, foot_sat, t), pw, BH)
    canvas = np.vstack([top, np.hstack([bl, br])])
    label(canvas, "Drone video", 12, 26); label(canvas, "Coverage - BEV", 12, toph + 26); label(canvas, "Coverage - Satellite", pw + 12, toph + 26)
    writer.write(canvas); t += 1.0 / FPS
cap.release(); writer.release()
print(f"wrote {out}  {round(T1 - T0, 1)}s")
