# Drone-video → 3D → satellite-map pipeline (local-GPU runbook / prompt)

This folder reproduces the whole pipeline on a **machine with a local NVIDIA GPU** (no RunPod):
a HUD-overlaid drone video of a parking lot → VGGT 3D reconstruction → north-up bird's-eye
map (BEV) → rough registration to a Google-satellite crop → an orbit demo and a 3-panel
"camera-coverage on the map" video.

> **Using this as a prompt for an agent (e.g. Claude Code) on the new machine:**
> "Set up and run the pipeline in `pipeline/` on my video `<path>` and satellite crop
> `<path>`. Follow README.md: do the per-video calibration (step 3) first by showing me a
> frame so I can confirm the crop/mask/segment/north, then run steps 4–8. It's a repetitive
> parking lot, so don't waste time on SIFT/LoFTR auto-matching — use the yellow-line
> registration (`align_yellow.py`) as in the README."

---

## 0. What you need
- Linux + **NVIDIA GPU** (CUDA). ~8–12 GB VRAM is plenty (VGGT-1B, ~36 frames @ 518px, a few seconds).
- Python 3.10–3.12 (torch wheels are reliable here; 3.14 also worked but with warnings).
- Two inputs:
  1. The **drone video** (with its burned-in HUD: a compass + an altitude readout).
  2. A **north-up satellite crop** of the same lot (screenshot from Google Maps satellite),
     PNG. The lot's distinctive static features (a yellow line / walkway, hatched zones, a
     canopy) must be visible.

## 1. Setup (once)
```bash
cd pipeline
python3 -m venv venv && . venv/bin/activate
# Torch with CUDA (pick the index matching your CUDA; cu124/cu126/cu128 all fine):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow opencv-python einops huggingface_hub safetensors trimesh
# VGGT model code (put on PYTHONPATH when running run_vggt.py):
git clone https://github.com/facebookresearch/vggt.git
```
First `run_vggt.py` downloads the VGGT-1B weights (~5 GB) from HuggingFace.

## 2. Pipeline at a glance
```
video ─▶ motion_profile.py ─▶ (pick segment: skip static start … until descent begins)
      ─▶ extract_frames.py  ─▶ frames/ (square-cropped, HUD-masked) + times.npy
      ─▶ run_vggt.py        ─▶ points.ply + cameras.npz          [GPU]
      ─▶ render_views.py    ─▶ view_xy/xz/yz.png  (sanity check)
      ─▶ north_bev.py       ─▶ north_bev.png (+ car-free north_bev_ground.png)
      ─▶ align_yellow.py    ─▶ M_yellow.npy + aligned_overlay.jpg (BEV→satellite)
      ─▶ render_orbit.py    ─▶ orbit.gif (3D demo)
      ─▶ footprint_3panel.py─▶ the final 3-panel coverage video
```

## 3. Per-video calibration (DO THIS FIRST — it's HUD/scene-specific)
Open one frame (`cv2`/any viewer) and set these, which are **tuned to the sample video** and
**will differ** for a new one:
- **`extract_frames.py` → `CROP`/`MASK`**: a SQUARE region of clean scene (square avoids
  VGGT's 518×518 aspect squish), plus black-out rectangles over any overlay inside it.
- **Segment [t0,t1]**: run `python motion_profile.py <video>` → low bars = static hover to
  **skip**; pick t0 at sustained motion. For t1, read the HUD **altitude**: stop where it
  starts **descending** (for the sample: t0≈20 s, t1≈85 s).
- **North convention** (`NORTH` env for `north_bev.py` & `footprint_3panel.py`): read the
  **compass** — which image direction is North? Sample video: N = image-right → `NORTH=x`
  (options: `x|-x|y|-y`).
- **`align_yellow.py` `SAT_ROI`**: box around the yellow/orange walkway in the satellite crop.
- **`footprint_3panel.py` `VIDEO_ROT`**: the sample video is saved portrait but is really
  landscape → `VIDEO_ROT=ccw`.

## 4–8. Run it
```bash
. venv/bin/activate
VID=/path/to/video.mp4 ; SAT=/path/to/satellite.png ; OUT=./out ; mkdir -p $OUT

# 4. frames (edit CROP/MASK + segment first)   t0 t1 num_frames
python extract_frames.py $VID $OUT/frames 20 85 36

# 5. reconstruct (GPU)
PYTHONPATH=./vggt python run_vggt.py $OUT/frames $OUT
python render_views.py $OUT/points.ply $OUT          # optional sanity check

# 6. north-up BEV (+ car-free)
NORTH=x python north_bev.py $OUT/points.ply $OUT/cameras.npz $OUT

# 7. register BEV → satellite (rough, yellow-line)
SAT_ROI="555 350 690 720" python align_yellow.py $OUT/north_bev.png $SAT $OUT
#   -> check aligned_overlay.jpg; adjust SAT_ROI / verify the yellow line found a good fit

# 8a. 3D orbit demo
python render_orbit.py $OUT/points.ply $OUT          # ZSIGN=1 TILT=72 PS=3 by default

# 8b. final 3-panel coverage video
NORTH=x VIDEO_ROT=ccw python footprint_3panel.py \
    $OUT/points.ply $OUT/cameras.npz $OUT/frames/times.npy \
    $OUT/north_bev.png $SAT $VID $OUT/footprint_3panel.mp4
```

## Gotchas & lessons (these cost real time — don't relearn them)
- **HUD overlay is static across frames** → a feature matcher treats it as rigid structure and
  corrupts poses/depth. Must crop + mask it out. (Square crop also fixes the aspect squish.)
- **VGGT resizes inputs to 518×518** (a square). Feed SQUARE crops so you don't distort aspect.
- **Registration is the hard part.** Generic matchers (SIFT, LoFTR, even with car-removal and
  masks) FAIL here — a parking lot is repetitive + the satellite is cross-domain (cars moved,
  different day). The reliable path is the **compass** (fixes rotation: north-up) + the
  **yellow-line fit** (`align_yellow.py`, fixes the residual). Still only ~stall-accurate.
  Proper upgrade: metric scale from altitude, a 2nd anchor (hatched zone), and an ECC refine.
- **Z-axis / orientation**: the orbit looked right with `ZSIGN=1`; point size `PS=3` fills gaps
  for a smooth surface.
- **Video saved portrait but is landscape** → rotate for display (`VIDEO_ROT=ccw`); this does
  NOT affect the reconstruction (VGGT is orientation-agnostic; the compass fixes north).
- **Sync**: tie footprints to the REAL per-frame timestamps (`times.npy`), not a uniform
  assumption from t=0 — the static-skip makes t0≠0.
- **numpy 2.x**: `ndarray.ptp()` was removed → use `np.ptp(arr)`. Always `np.linalg.svd(X,
  full_matrices=False)` on tall matrices (full U is N×N and hangs/OOMs).
- **Output codec**: these scripts write `mp4v` (works offline, no H.264 needed). Some browsers
  won't inline-play it → download to view, or transcode with ffmpeg if available.

## Not needed (dead-ends we tried, documented so you don't repeat them)
SIFT (`cv2`) and LoFTR/RoMa learned matching for BEV↔satellite (both gave degenerate fits on
this repetitive scene); satellite car-removal by inpainting + LoFTR masks (didn't help the
match); a three.js web point-cloud viewer. The compass+yellow approach above replaced all of it.

## Next milestones (not yet built)
- Tighten registration (altitude metric scale + hatch anchor + ECC).
- **Empty-spot detection** (the real goal — find free stalls), then project detected spots
  onto the map with the same transform.
- Optionally OCR the HUD (altitude + heading) to make the reconstruction metric & north-aligned
  automatically instead of reading the compass by eye.
```
