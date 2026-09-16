"""
Thin wrapper around the COLMAP CLI to run sparse SfM on a folder of images.

Requires a working `colmap` binary on PATH (not available in this dev
container -- install separately: https://colmap.github.io/install.html).

Usage:
    python run_colmap.py --image_dir /path/to/images --workspace /path/to/workspace
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"[run_colmap] $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[run_colmap] command failed with code {result.returncode}: {' '.join(cmd)}")


def run_sfm(image_dir: Path, workspace: Path, camera_model: str = "OPENCV",
            single_camera: bool = True, matcher: str = "exhaustive") -> Path:
    """
    Runs COLMAP feature extraction -> matching -> incremental mapping.

    single_camera=True is correct for our use case: one physical phone camera
    used for the whole capture, so all images share intrinsics. This is
    important -- letting COLMAP estimate per-image intrinsics independently
    on a small/low-parallax capture (a limb, not a building) is a common
    source of scale/geometry instability.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ])

    matcher_cmd = {
        "exhaustive": ["colmap", "exhaustive_matcher"],
        "sequential": ["colmap", "sequential_matcher"],
    }[matcher]
    run(matcher_cmd + [
        "--database_path", str(database_path),
        "--SiftMatching.guided_matching", "1",
    ])

    run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        # Tighter thresholds than COLMAP defaults: a limb capture has low
        # parallax and repetitive low-texture surface, which produces more
        # spurious matches than a textured scene. Being stricter here trades
        # completeness for not silently accepting bad triangulations.
        "--Mapper.filter_max_reproj_error", "2.0",
        "--Mapper.tri_min_angle", "2.0",
    ])

    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        sys.exit(
            "[run_colmap] no reconstruction produced (sparse/0 missing). "
            "Likely causes for this use case: too few images, insufficient "
            "overlap between viewpoints, or too little texture/features on "
            "the limb surface for reliable matching."
        )
    return model_dir


def undistort(image_dir: Path, model_dir: Path, workspace: Path) -> Path:
    """Undistort images + export in a format the downstream converter expects."""
    dense_dir = workspace / "dense"
    run([
        "colmap", "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ])
    return dense_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--camera_model", default="OPENCV")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    args = parser.parse_args()

    model_dir = run_sfm(args.image_dir, args.workspace, args.camera_model, matcher=args.matcher)
    dense_dir = undistort(args.image_dir, model_dir, args.workspace)
    print(f"[run_colmap] sparse model: {model_dir}")
    print(f"[run_colmap] undistorted images + poses for downstream use: {dense_dir}")
