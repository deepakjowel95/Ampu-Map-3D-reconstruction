# Residual-Limb Reconstruction Baseline: COLMAP → SDF (NeuS-style)

## What this is

A baseline pipeline for reconstructing a 3D surface of a residual limb from a
set of smartphone images:

```
images (+ optional ArUco marker)
        │
        ▼
   COLMAP SfM   ──►  sparse point cloud + camera poses (up to unknown global scale)
        │
        ▼
 colmap_to_neus.py ──► NeuS-format cameras_sphere.npz (scene normalized to unit sphere)
        │
        ▼
   SDF training (NeuS/VolSDF-style: SDFNetwork + RenderingNetwork + volume renderer)
        │
        ▼
 marching_cubes ──► mesh (unit-sphere scale)
        │
        ▼
 aruco_scale_calibration.py ──► mesh rescaled to real-world millimeters
```

## Why SDF instead of density-field NeRF

Per prior discussion: vanilla NeRF represents geometry as a volumetric density
field, which has no surface inductive bias and degrades on low-texture,
smooth organic surfaces (skin). NeuS/VolSDF instead model geometry as a
**signed distance function** and derive volume-rendering weights from it, which
gives a much stronger surface prior — closer to what MVS assumes implicitly,
better matched to a solid, closed limb geometry.

## What is validated in this baseline vs. what is not

This environment has no GPU and no COLMAP binary available (network is
restricted to package registries only), so:

- **Validated here**: network architectures (`fields.py`), volume-rendering
  math (`renderer.py`), COLMAP→NeuS conversion logic, and the training loop
  wiring — all exercised end-to-end with synthetic dummy data via
  `scripts/sanity_check.py` to catch shape/logic bugs before you spend GPU
  time on it.
- **Not validated here**: actual COLMAP reconstruction quality, actual SDF
  convergence/mesh accuracy on real limb photos, actual metric error against
  ground truth. Those require a GPU box and real (or phantom) capture data —
  do that next, not in this container.

## The metric-accuracy question is still open

This baseline gets you a working pipeline. It does **not** by itself solve the
"improve the metric" problem discussed earlier. The two concrete levers left
to actually pull, in priority order:

1. **Fix global scale.** COLMAP (monocular, no metric input) recovers geometry
   only up to an unknown scale factor. `aruco_scale_calibration.py` is a stub
   for resolving this using a marker of known physical size in the capture —
   without this, every mm-level claim you make about the output is
   meaningless.
2. **Depth-supervise the SDF training**, using COLMAP's sparse point cloud
   (`points3D`) as in DS-NeRF, so the surface network is anchored to
   metrically-consistent (if sparse) 3D points rather than photometric loss
   alone. Hook point is `SDFTrainer.get_depth_loss()` in `exp_runner.py` —
   currently a stub with a TODO, deliberately left for you to implement and
   ablate, since that's the actual research contribution, not something to
   hand you pre-solved.

## Directory layout

```
colmap_pipeline/
  run_colmap.py             # wraps colmap CLI: feature_extractor, matcher, mapper
  colmap_to_neus.py         # COLMAP output -> cameras_sphere.npz
  aruco_scale_calibration.py# recover metric scale from a known-size marker
sdf_reconstruction/
  models/fields.py          # SDFNetwork, RenderingNetwork, SingleVarianceNetwork
  models/renderer.py        # NeuS-style volume renderer (SDF -> rendering weights)
  dataset.py                # loads cameras_sphere.npz + images, generates rays
  exp_runner.py             # training loop
  confs/base.conf           # hyperparameters
scripts/
  extract_mesh.py           # marching cubes on trained SDF -> .ply mesh
  sanity_check.py           # CPU, dummy-data smoke test of the whole stack
```

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install colmap        # or build from source; not available in this container
python scripts/sanity_check.py     # verify the code runs before touching real data/GPU
```
