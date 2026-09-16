"""
Smoke test: exercises dataset loading, one training step, and mesh
extraction end-to-end using synthetic dummy data (no COLMAP, no GPU, no
real images required). This does NOT validate reconstruction quality --
only that the code runs and shapes/gradients are wired correctly, so you
don't burn GPU hours discovering a bug in the plumbing.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sdf_reconstruction"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import SceneDataset
from models.fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer
from scripts.extract_mesh import extract_mesh


def make_synthetic_scene(root: Path, n_images=4, H=64, W=64):
    """A handful of cameras on a ring pointed at the origin, with random
    RGB images -- enough to exercise the data path, not enough to
    reconstruct anything meaningful."""
    (root / "image").mkdir(parents=True)
    world_mats, scale_mats = {}, {}
    K = np.array([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    radius = 3.0
    for i in range(n_images):
        theta = 2 * np.pi * i / n_images
        cam_pos = radius * np.array([np.cos(theta), 0, np.sin(theta)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        R_w2c = np.stack([right, true_up, -forward], axis=0)  # world->cam rows
        t = -R_w2c @ cam_pos
        Rt = np.concatenate([R_w2c, t.reshape(3, 1)], axis=1)
        P = np.eye(4)
        P[:3, :4] = K @ Rt
        world_mats[i] = P

        img = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(root / "image" / f"{i:03d}.png"), img)

    scale_mat = np.eye(4)  # identity: synthetic scene already ~unit scale
    save_dict = {}
    for i in range(n_images):
        save_dict[f"world_mat_{i}"] = world_mats[i]
        save_dict[f"scale_mat_{i}"] = scale_mat
    np.savez(root / "cameras_sphere.npz", **save_dict)
    np.save(root / "sparse_points.npy", (np.random.rand(200, 3) - 0.5) * 0.5)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sdf_sanity_"))
    try:
        print(f"[sanity_check] synthetic scene at {tmp}")
        make_synthetic_scene(tmp)

        print("[sanity_check] loading dataset...")
        dataset = SceneDataset(tmp)
        assert dataset.n_images == 4
        assert dataset.images.shape == (4, 64, 64, 3)
        print(f"[sanity_check] OK -- {dataset.n_images} images, "
              f"{dataset.sparse_points.shape[0]} sparse points loaded")

        print("[sanity_check] building networks...")
        sdf_network = SDFNetwork(d_out=257, n_layers=4, d_hidden=64, multires=4)
        color_network = RenderingNetwork(d_feature=256, d_hidden=64, n_layers=2, multires_view=2)
        deviation_network = SingleVarianceNetwork()
        renderer = NeuSRenderer(sdf_network, deviation_network, color_network,
                                 n_samples=16, n_importance=16, up_sample_steps=2)

        print("[sanity_check] running one training step (forward + backward)...")
        rays_o, rays_d, color, mask = dataset.gen_random_rays_at(0, batch_size=32)
        near, far = dataset.near_far_from_sphere(rays_o, rays_d)
        out = renderer.render(rays_o, rays_d, near, far)
        assert out["color"].shape == (32, 3)
        assert out["depth"].shape == (32, 1)

        params = list(sdf_network.parameters()) + list(color_network.parameters()) + list(deviation_network.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)
        loss = torch.nn.functional.l1_loss(out["color"], color) + 0.1 * out["gradient_error"]
        optimizer.zero_grad()
        loss.backward()
        # check gradients actually flowed into the SDF network, not just the color head
        grad_norms = [p.grad.norm().item() for p in sdf_network.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and max(grad_norms) > 0, "no gradient reached SDFNetwork -- rendering graph is disconnected"
        optimizer.step()
        print(f"[sanity_check] OK -- loss={loss.item():.4f}, "
              f"max SDF grad norm={max(grad_norms):.6f} (nonzero => gradient path is connected)")

        print("[sanity_check] extracting mesh at low resolution...")
        mesh = extract_mesh(sdf_network, resolution=24, bound=1.0)
        print(f"[sanity_check] OK -- mesh has {len(mesh.vertices)} verts "
              f"(untrained network, so this checks marching-cubes plumbing "
              f"only, not reconstruction quality)")

        print("\n[sanity_check] ALL CHECKS PASSED -- plumbing is sound. "
              "This does not validate reconstruction accuracy on real data.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
