"""
Loads a NeuS-format scene (images/ + cameras_sphere.npz produced by
colmap_pipeline/colmap_to_neus.py) and generates training rays.

world_mat_i = K @ [R|t] (camera projection in COLMAP scale)
scale_mat_i = similarity transform, COLMAP scale -> unit sphere (training scale)
Combined and decomposed via RQ to recover per-image (intrinsics, pose) in
the *normalized* (unit-sphere) coordinate frame NeuS trains in.
"""
from pathlib import Path

import cv2
import numpy as np
import torch


def load_K_Rt_from_P(P: np.ndarray):
    """Decompose a 3x4 projection matrix into intrinsics K and pose [R|t]
    via RQ decomposition (standard OpenCV/NeuS convention)."""
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K.astype(np.float32), pose


class SceneDataset:
    def __init__(self, data_dir: Path, image_glob: str = "*.png"):
        self.data_dir = Path(data_dir)
        camera_data = np.load(self.data_dir / "cameras_sphere.npz")
        image_paths = sorted((self.data_dir / "image").glob(image_glob))
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.data_dir / 'image'} matching '{image_glob}'. "
                "Expected the undistorted images from run_colmap.py's "
                "image_undistorter step, copied into <data_dir>/image/."
            )
        self.n_images = len(image_paths)

        images, masks = [], []
        intrinsics_all, pose_all = [], []
        for i, path in enumerate(image_paths):
            world_mat = camera_data[f"world_mat_{i}"]
            scale_mat = camera_data[f"scale_mat_{i}"]
            P = (world_mat @ scale_mat)[:3, :4]
            K, pose = load_K_Rt_from_P(P)

            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            images.append(img)

            mask_path = self.data_dir / "mask" / path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            else:
                mask = np.ones(img.shape[:2], dtype=np.float32)
            masks.append(mask)

            intrinsics_all.append(K)
            pose_all.append(pose)

        self.images = torch.from_numpy(np.stack(images))         # [N, H, W, 3]
        self.masks = torch.from_numpy(np.stack(masks))           # [N, H, W]
        self.intrinsics_all = torch.from_numpy(np.stack(intrinsics_all))
        self.pose_all = torch.from_numpy(np.stack(pose_all))
        self.H, self.W = self.images.shape[1], self.images.shape[2]

        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)

        if (self.data_dir / "sparse_points.npy").exists():
            self.sparse_points = torch.from_numpy(
                np.load(self.data_dir / "sparse_points.npy")
            ).float()
        else:
            self.sparse_points = None

    def gen_random_rays_at(self, img_idx: int, batch_size: int):
        """Sample `batch_size` random pixel rays from image `img_idx`."""
        pixels_x = torch.randint(0, self.W, (batch_size,))
        pixels_y = torch.randint(0, self.H, (batch_size,))
        color = self.images[img_idx][pixels_y, pixels_x]
        mask = self.masks[img_idx][pixels_y, pixels_x]

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_x)], dim=-1).float()
        p = p @ self.intrinsics_all_inv[img_idx, :3, :3].T
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_d = rays_d @ self.pose_all[img_idx, :3, :3].T
        rays_o = self.pose_all[img_idx, :3, 3].expand(rays_d.shape)

        return rays_o, rays_d, color, mask[..., None]

    def near_far_from_sphere(self, rays_o, rays_d):
        """Ray-unit-sphere intersection: since the scene is normalized to
        fit inside the unit sphere, near/far bounds are just the sphere
        intersection, not a dataset-specific guess."""
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return torch.clamp_min(near, 1e-3), far
