"""
Extract a triangle mesh from a trained SDFNetwork via marching cubes, then
map it back from unit-sphere training coordinates to COLMAP scale using
`scale_mat`.

NOTE: this mesh is still in *COLMAP-arbitrary* units after the scale_mat
un-normalization, not millimeters -- run aruco_scale_calibration.py's
scale_factor on top of this output to get real-world units. Reporting a
number from this script's output as a clinical measurement without that
step is exactly the metric-vs-training-scale conflation flagged earlier.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdf_reconstruction"))
from models.fields import SDFNetwork


def extract_mesh(sdf_network: SDFNetwork, resolution: int, bound: float = 1.0) -> trimesh.Trimesh:
    grid = np.linspace(-bound, bound, resolution)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    sdf_vals = np.zeros(pts.shape[0], dtype=np.float32)
    chunk = 64 ** 3
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            batch = torch.from_numpy(pts[i:i + chunk]).float()
            sdf_vals[i:i + chunk] = sdf_network.sdf(batch).squeeze(-1).numpy()
    sdf_vals = sdf_vals.reshape(resolution, resolution, resolution)

    if sdf_vals.min() > 0 or sdf_vals.max() < 0:
        raise RuntimeError(
            "SDF grid never crosses zero -- no surface found inside the "
            f"[-{bound}, {bound}] cube. Likely an undertrained network or "
            "the object doesn't fit inside this bound; check scale_mat's "
            "padding in colmap_to_neus.py."
        )

    verts, faces, normals, _ = measure.marching_cubes(sdf_vals, level=0.0)
    # rescale from voxel index space back to [-bound, bound]
    verts = verts / (resolution - 1) * (2 * bound) - bound
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


def to_colmap_scale(mesh: trimesh.Trimesh, scale_mat: np.ndarray) -> trimesh.Trimesh:
    verts_h = np.concatenate([mesh.vertices, np.ones((len(mesh.vertices), 1))], axis=1)
    verts_colmap = (scale_mat @ verts_h.T).T[:, :3]
    return trimesh.Trimesh(vertices=verts_colmap, faces=mesh.faces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--cameras_npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--conf", type=Path, default=Path(__file__).parent.parent / "sdf_reconstruction/confs/base.conf")
    args = parser.parse_args()

    from pyhocon import ConfigFactory
    conf = ConfigFactory.parse_file(str(args.conf))
    sdf_network = SDFNetwork(**conf["model.sdf_network"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sdf_network.load_state_dict(ckpt["sdf_network"])
    sdf_network.eval()

    mesh = extract_mesh(sdf_network, args.resolution)

    cameras = np.load(args.cameras_npz)
    scale_mat = cameras["scale_mat_0"]  # same scale_mat for every image, see colmap_to_neus.py
    mesh_colmap_scale = to_colmap_scale(mesh, scale_mat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh_colmap_scale.export(args.out)
    print(f"[extract_mesh] wrote {len(mesh_colmap_scale.vertices)} verts, "
          f"{len(mesh_colmap_scale.faces)} faces to {args.out}")
    print("[extract_mesh] units are COLMAP-arbitrary, NOT millimeters -- "
          "apply aruco_scale_calibration.py's scale_factor before reporting "
          "any dimensional measurement.")
