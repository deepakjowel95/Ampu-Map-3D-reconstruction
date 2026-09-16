"""
Recover the real-world metric scale factor of a COLMAP/NeuS reconstruction
using an ArUco marker of known physical size placed in the capture.

This is the mandatory step before any mm-level claim about the output mesh
means anything -- COLMAP alone recovers geometry only up to an unknown
scale (monocular reconstruction has no absolute scale reference).

Method: detect the marker in >=2 images with known COLMAP poses, triangulate
its 4 corners in COLMAP-space, compute the corner-to-corner distance in
COLMAP units, and compare to the known real-world marker size (in mm) to get
a single scalar `scale_factor`. Apply that factor (not the unit-sphere
scale_mat, which is for training normalization only) to the final extracted
mesh.
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICT = cv2.aruco.DICT_4X4_50


def detect_marker_corners(image_path: Path, marker_id: int) -> np.ndarray | None:
    """Returns the 4 pixel-space corners (4,2) of the given marker id, or None."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return None
    ids = ids.flatten()
    if marker_id not in ids:
        return None
    idx = int(np.where(ids == marker_id)[0][0])
    return corners[idx].reshape(4, 2)


def triangulate_point(P1: np.ndarray, pt1: np.ndarray, P2: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation of a single point from two views."""
    X = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
    return (X[:3] / X[3]).flatten()


def estimate_scale_factor(
    world_mats: dict[str, np.ndarray],
    image_dir: Path,
    marker_id: int,
    marker_size_mm: float,
) -> float:
    """
    Returns: scale_factor such that
        real_world_mm = colmap_units * scale_factor
    """
    detections = {}
    for name in world_mats:
        corners = detect_marker_corners(image_dir / name, marker_id)
        if corners is not None:
            detections[name] = corners

    if len(detections) < 2:
        raise RuntimeError(
            f"Marker {marker_id} detected in only {len(detections)} image(s); "
            "need >= 2 views with different viewpoints to triangulate. "
            "This is a capture-protocol requirement, not something this "
            "script can work around -- the marker must stay visible and "
            "flat-on-camera in at least two frames of the orbit."
        )

    name_a, name_b = list(itertools.combinations(detections.keys(), 2))[0]
    P1, P2 = world_mats[name_a][:3, :4], world_mats[name_b][:3, :4]
    corners_3d = np.stack([
        triangulate_point(P1, detections[name_a][k], P2, detections[name_b][k])
        for k in range(4)
    ])

    # Marker is a square: use all 4 side lengths, average for robustness
    # against triangulation noise on any single edge.
    side_lengths_colmap = [
        np.linalg.norm(corners_3d[k] - corners_3d[(k + 1) % 4]) for k in range(4)
    ]
    mean_side_colmap = float(np.mean(side_lengths_colmap))
    spread = float(np.std(side_lengths_colmap) / mean_side_colmap)
    if spread > 0.1:
        print(
            f"[aruco_scale_calibration] WARNING: side-length spread {spread:.1%} "
            "across the 4 marker edges -- triangulation looks unreliable "
            "(check marker visibility/angle in the two chosen views)."
        )

    return marker_size_mm / mean_side_colmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras_npz", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--marker_id", type=int, default=0)
    parser.add_argument("--marker_size_mm", type=float, required=True)
    args = parser.parse_args()

    data = np.load(args.cameras_npz)
    n_images = sum(1 for k in data.files if k.startswith("world_mat_"))
    # NOTE: relies on cameras_sphere.npz having been built with images sorted
    # by filename, same convention as colmap_to_neus.py.
    image_names = sorted(p.name for p in args.image_dir.iterdir())
    world_mats = {image_names[i]: data[f"world_mat_{i}"] for i in range(n_images)}

    scale = estimate_scale_factor(world_mats, args.image_dir, args.marker_id, args.marker_size_mm)
    print(f"[aruco_scale_calibration] scale_factor = {scale:.6f} "
          f"(multiply COLMAP-unit mesh coordinates by this to get millimeters)")
