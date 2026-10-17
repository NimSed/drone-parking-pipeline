#!/usr/bin/env python3
"""Feed-forward 3D reconstruction with VGGT (runs on a LOCAL GPU).
Input: a folder of preprocessed frames. Output: points.ply + cameras.npz.

Requires: torch(+CUDA), and the VGGT repo on PYTHONPATH (git clone facebookresearch/vggt).
Usage:  PYTHONPATH=/path/to/vggt python run_vggt.py <frames_dir> <out_dir>
Env:    CONF_PCT=50   # drop the lowest X% confidence points
"""
import os, glob, sys, numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IMG_DIR, OUT_DIR = sys.argv[1], sys.argv[2]
CONF_PCT = float(os.environ.get("CONF_PCT", "50"))
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) + glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(paths) >= 2, "need >=2 frames"
print(f"[vggt] {len(paths)} frames | loading VGGT-1B (downloads ~5GB on first run)...", flush=True)
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
images = load_and_preprocess_images(paths).to(device)        # (N,3,518,518)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    pred = model(images)

wp   = pred["world_points"][0].float().cpu().numpy()          # (N,H,W,3)
conf = pred["world_points_conf"][0].float().cpu().numpy()
imgs = images.cpu().numpy().transpose(0, 2, 3, 1)             # (N,H,W,3) in [0,1]
pts  = wp.reshape(-1, 3)
cols = (imgs.reshape(-1, 3) * 255).astype(np.uint8)
c    = conf.reshape(-1)
thr  = np.percentile(c, CONF_PCT)
keep = (c >= thr) & np.isfinite(pts).all(1)
pts, cols = pts[keep], cols[keep]
print(f"[vggt] kept {keep.sum()}/{keep.size} pts (conf>= {thr:.3f})", flush=True)

ply = os.path.join(OUT_DIR, "points.ply")
with open(ply, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, col in zip(pts, cols):
        f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {col[0]} {col[1]} {col[2]}\n")

extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
np.savez(os.path.join(OUT_DIR, "cameras.npz"),
         extrinsics=extr[0].float().cpu().numpy(),      # (N,3,4) world->cam (OpenCV)
         intrinsics=intr[0].float().cpu().numpy(),      # (N,3,3) for the 518x518 input
         image_paths=paths)
print(f"[vggt] wrote {ply} ({len(pts)} pts) + cameras.npz", flush=True)
