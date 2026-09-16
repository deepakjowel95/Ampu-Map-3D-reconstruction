"""
Convert a COLMAP sparse reconstruction into the NeuS `cameras_sphere.npz`
format: per-image `world_mat_%d` (3x4 projection matrix = K @ [R|t]) and
`scale_mat_%d` (4x4 similarity transform mapping the COLMAP-scale scene into
a unit sphere centered on the reconstructed object).

Also dumps the sparse point cloud as `sparse_points.npy` -- this is the
signal `exp_runner.py`'s depth-supervision hook should consume (see
README: "Depth-supervise the SDF training").

COLMAP scale is arbitrary (monocular reconstruction, no metric reference).
The unit-sphere normalization here is *for training stability only* -- it is
NOT metric. Recovering true millimeters requires
`aruco_scale_calibration.py` downstream of mesh extraction.
"""
import argparse
from pathlib import Path

import numpy as np

try:
    import pycolmap
except ImportError:  # pragma: no cover - only needed for real COLMAP runs
    pycolmap = None


def load_reconstruction(model_dir: Path):
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required to read a real COLMAP model. "
            "Install it with `pip install pycolmap`, or use "
            "build_from_arrays() directly for testing without COLMAP."
        )
    return pycolmap.Reconstruction(str(model_dir))


def intrinsics_matrix(camera) -> np.ndarray:
    """Build a 3x3 K matrix from a pycolmap Camera (handles common models)."""
    params = camera.params
    model = camera.model.name
    K = np.eye(3)
    if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"Unsupported camera model for this baseline: {model}")
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    return K


def build_world_mats(recon) -> dict[str, np.ndarray]:
    """world_mat[i] = K @ [R | t], the 3x4 projection matrix for image i."""
    world_mats = {}
    for image_id, image in recon.images.items():
        camera = recon.cameras[image.camera_id]
        K = intrinsics_matrix(camera)
        R = image.cam_from_world.rotation.matrix()
        t = image.cam_from_world.translation
        Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)  # 3x4
        world_mats[image.name] = K @ Rt
    return world_mats


def build_scale_mat(recon, padding: float = 1.1) -> np.ndarray:
    """
    4x4 similarity transform that maps the object's bounding sphere (from
    the sparse point cloud) onto the unit sphere. `padding` keeps a margin
    so the object doesn't touch the sphere boundary during training.
    """
    points = np.array([p.xyz for p in recon.points3D.values()])
    if len(points) < 10:
        raise RuntimeError(
            f"Only {len(points)} sparse points -- too few to estimate scene "
            "extent reliably. This usually means the COLMAP reconstruction "
            "itself is bad (see run_colmap.py troubleshooting note), not a "
            "bug in this converter."
        )
    center = points.mean(axis=0)
    radius = np.linalg.norm(points - center, axis=1).max() * padding

    scale_mat = np.eye(4)
    scale_mat[:3, :3] *= radius
    scale_mat[:3, 3] = center
    return scale_mat, points


def convert(model_dir: Path, out_dir: Path) -> None:
    recon = load_reconstruction(model_dir)
    world_mats = build_world_mats(recon)
    scale_mat, sparse_points = build_scale_mat(recon)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    # Sort by filename so image index i in cameras_sphere.npz matches the
    # i-th image in the undistorted image directory NeuS's dataset.py expects.
    for i, name in enumerate(sorted(world_mats.keys())):
        P = np.eye(4)
        P[:3, :4] = world_mats[name]
        save_dict[f"world_mat_{i}"] = P
        save_dict[f"scale_mat_{i}"] = scale_mat

    np.savez(out_dir / "cameras_sphere.npz", **save_dict)
    np.save(out_dir / "sparse_points.npy", sparse_points)
    print(f"[colmap_to_neus] wrote {len(world_mats)} cameras to "
          f"{out_dir / 'cameras_sphere.npz'}")
    print(f"[colmap_to_neus] wrote {len(sparse_points)} sparse points to "
          f"{out_dir / 'sparse_points.npy'} (use for depth supervision)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True,
                         help="COLMAP sparse model dir, e.g. workspace/sparse/0")
    parser.add_argument("--out_dir", type=Path, required=True,
                         help="Where to write cameras_sphere.npz")
    args = parser.parse_args()
    convert(args.model_dir, args.out_dir)
