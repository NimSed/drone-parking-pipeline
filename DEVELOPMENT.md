# Drone → 3D → Satellite parking pipeline — development prompt & journey

**Read this first.** It is both (a) the *story and rationale* of how this pipeline was built —
so you know why each step exists and, just as important, what **not** to retry — and (b) the
instructions to fetch, run, and extend it on a machine with a **local NVIDIA GPU**.

> **The code lives in this repository.** On a new machine, clone it (see §6). This document is
> self-contained: even without the code in front of you, it captures the full flow, the
> exceptions/expectations, and the next directions, so you can keep developing.

---

## 1. The goal
Big picture: **find free parking spots from a drone flyover and route a driver to them.**
This pipeline is the mapping backbone for that:

1. Reconstruct the parking lot in 3D from a HUD-overlaid drone video (feed-forward, fast).
2. Build a **north-up bird's-eye (BEV)** map of it.
3. **Register** that BEV to a north-up **Google-satellite crop** of the same lot.
4. Visualize **camera coverage** (which areas the flight saw) on both the BEV and the map.

**Not yet built (next milestones):** empty-stall *detection*, then projecting detected free
spots onto the map with the same transform; and optionally OCR-ing the HUD to make the
reconstruction metric + north-aligned automatically.

## 2. Final pipeline (data flow)
```
video ─▶ motion_profile.py   pick segment: SKIP the static hover, stop where DESCENT begins
      ─▶ extract_frames.py   sample N frames, square-crop, black-out the HUD, save times.npy
      ─▶ run_vggt.py  [GPU]  VGGT-1B feed-forward -> points.ply + cameras.npz (poses+intrinsics)
      ─▶ render_views.py     3 orthographic sanity views
      ─▶ north_bev.py        PCA ground plane -> BEV; compass heading -> rotate NORTH-UP; +car-free
      ─▶ align_yellow.py     register BEV->satellite by the yellow line <-> walkway (similarity)
      ─▶ render_orbit.py     3D orbit demo GIF
      ─▶ footprint_3panel.py video(top) + BEV-coverage + satellite-coverage, persistent, synced
```

## 3. Why each step is the way it is (decisions & dead-ends)

**Model = VGGT (feed-forward 3D).** Chosen because it reconstructs camera poses + dense points
from unposed images in one forward pass (no slow bundle adjustment) — "drone-friendly." Concern:
would it handle near-nadir *top-down* views (under-represented in its training)? Tested on this
footage: **yes**, it reconstructs the lot coherently. It needs real parallax, so a pure static
hover is degenerate — hence we skip the static opening.

**The HUD overlay is the first trap.** The video is a screen-recording with a burned-in HUD
(compass, telemetry, buttons). Those elements are **static in screen space across all frames**,
so any geometry method treats them as rigid foreground and corrupts poses/depth. → We **crop +
mask** them out. And because **VGGT resizes inputs to 518×518 (square)**, we take a **square**
center crop so the aspect isn't squished.

**Segment selection.** `motion_profile.py` (frame-to-frame change) reveals the static hover at
the start → skip it. The end is where the drone starts **descending to land** — read from the
HUD **altitude** (it plateaus ~19 m, then drops). For the sample video the flying segment is
≈ **20–85 s**; we sample **36 frames** evenly and **save each frame's real timestamp**
(`times.npy`) — crucial for sync later.

**BEV + north-up.** The reconstruction frame is arbitrary (up-to-scale, no north). We fit the
ground plane with **PCA** (2 in-plane axes + normal) and project orthographically = BEV. To make
it north-up (to match a north-up satellite) we use the **compass**: read which *image* direction
is north (here **N = image-right** = the camera +x axis), transform that through the poses to get
north in the reconstruction frame, and rotate the BEV so north points up.

**Registration — the hard part, and the biggest lesson.** Aligning the BEV to the satellite by
*appearance* repeatedly failed:
- **SIFT** → degenerate homography. **LoFTR / learned matching** → a handful of inliers,
  degenerate fit, even after **removing the cars** (inpaint) and using **feature masks**.
- Hand-picking correspondences by eye (VLM) → **mutually inconsistent** (a self-intersecting
  "bowtie" transform), which *proved* the match was ambiguous, not just hard.
- **Root cause:** a parking lot is **highly repetitive** (every stall looks alike) and the
  satellite is **cross-domain** (shot a different day — cars moved, lighting/season differ). The
  problem is **under-constrained from appearance alone.**
- **What works:** add *external* constraints. The **compass** kills the rotation ambiguity
  (north-up). Then a one-feature fit — the **thick yellow line (BEV) ↔ the orange walkway
  (satellite)** — fixes the residual rotation + scale + translation (a similarity). This is the
  pragmatic, **non-final** registration: roughly stall-accurate. Proper upgrade: **metric scale
  from the HUD altitude + a second anchor (the hatched zone) + an ECC intensity refine.**

**Footprints & coverage.** Each frame's ground footprint = ray-cast the image corners through
`K` + pose onto the ground plane (clamp the near-horizon blow-ups of oblique frames). Project via
the poses (BEV) and `M` (satellite). The final video shows **persistent/accumulating** coverage.

**Sync.** Tie footprints to the **real per-frame timestamps** (`times.npy`), not a uniform
assumption from t=0 — the static-skip means t₀ ≠ 0. (An early version assumed uniform-from-0 and
the sync was visibly wrong.)

**Orbit demo.** Orthographic spin. Flip z (`ZSIGN=1`) so it reads right-side-up; **enlarge the
points** (`PS≈3`) to fill gaps into a continuous surface; tilt/zoom/speed are tunable.

## 4. Gotchas (these cost real time — don't relearn them)
- **Square crop** for VGGT (avoids the 518² aspect squish); and **mask the HUD** (static overlay
  corrupts matching).
- **Registration:** do NOT reach for SIFT/LoFTR/RoMa on this repetitive cross-domain scene —
  they produce confident-but-wrong fits. Use **compass + yellow-line**.
- **Orientation:** orbit looked right with `ZSIGN=1`; the sample video is saved *portrait* but is
  really *landscape* → rotate for display (`VIDEO_ROT=ccw`); this does **not** affect the
  reconstruction (VGGT is orientation-agnostic; the compass fixes north).
- **Sync** via real timestamps, not uniform-from-0.
- **numpy 2.x:** `ndarray.ptp()` was removed → `np.ptp(arr)`. Always `np.linalg.svd(X,
  full_matrices=False)` on tall matrices (full U is N×N → hang/OOM).
- **Output codec:** scripts write `mp4v` (no H.264 dependency). Some browsers won't inline-play
  it → download, or transcode with ffmpeg if present.
- Up-to-scale + arbitrary orientation are inherent to vision-only reconstruction; metric scale
  and north come from the **HUD** (altitude, compass), not VGGT.

## 5. Dead-ends (documented so you don't repeat them)
SIFT and LoFTR/RoMa for BEV↔satellite; satellite **car-removal by inpainting** + LoFTR masks (to
"clean" the match — didn't help); a **three.js web point-cloud viewer** (didn't render for the
user). The compass + yellow-line approach replaced all of these.

## 6. Get the code & run it
```bash
git clone <THIS_REPO_URL> drone-pipeline && cd drone-pipeline
python3 -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install numpy pillow opencv-python einops huggingface_hub safetensors trimesh
git clone https://github.com/facebookresearch/vggt.git                 # VGGT model code
# first run downloads VGGT-1B weights (~5 GB) from HuggingFace
```
Then **per video** (see README.md for exact commands):
1. **Calibrate (HUD/scene-specific):** open a frame; set `CROP`/`MASK` (square clean region +
   overlay boxes) in `extract_frames.py`; run `motion_profile.py` + read the HUD altitude to pick
   the segment `[t0,t1]`; read the **compass** for `NORTH` (x|-x|y|-y); set `SAT_ROI` (box around
   the walkway) and `VIDEO_ROT` for `align_yellow.py`/`footprint_3panel.py`.
2. Run: `extract_frames → run_vggt → render_views → north_bev → align_yellow → render_orbit →
   footprint_3panel` (commands in README.md §4–8).

## 7. Expectations & limitations
- The registration is **rough (≈stall-level)**, by design — it's the quick method. Everything up
  to the north-up BEV is general; only `align_yellow.py` is scene-specific.
- It assumes a **distinctive shared feature** exists in both BEV and satellite (here the yellow
  line ↔ walkway). A lot without such an anchor needs that step adapted.
- All params in step 1 are **per-video** (different HUD/framing/lot). The *code* is reusable; the
  *parameters* change.

## 8. Next development directions
1. **Tighten registration:** metric scale from HUD altitude + a 2nd anchor (hatched zone) + ECC
   refine → stall-precise coverage.
2. **Empty-stall detection** (the real goal) → project detected free spots onto the map via `M`.
3. **OCR the HUD** (altitude + heading) to make the reconstruction metric & north-aligned
   automatically instead of reading the compass by eye.
