"""
Training loop for COLMAP + SDF (NeuS-style) reconstruction.

Losses:
  - color_loss: L1 photometric loss against ground-truth pixels
  - mask_loss:  BCE between rendered opacity and the (optional) foreground mask
  - eikonal (igr) loss: regularizes |grad SDF| ~ 1 (computed in renderer.py)
  - depth_loss: STUBBED. This is the hook for the "improve the metric"
    work discussed -- supervising rendered depth against COLMAP's sparse
    point cloud (DS-NeRF-style), which anchors the surface to something
    metrically consistent rather than photometric loss alone. Deliberately
    left unimplemented; see README.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SceneDataset
from models.fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer


class SDFTrainer:
    def __init__(self, conf_path: Path, data_dir: Path, exp_dir: Path):
        self.conf = ConfigFactory.parse_file(str(conf_path))
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = SceneDataset(data_dir)

        sdf_conf = self.conf["model.sdf_network"]
        self.sdf_network = SDFNetwork(**sdf_conf)

        render_conf = self.conf["model.rendering_network"]
        self.color_network = RenderingNetwork(**render_conf)

        self.deviation_network = SingleVarianceNetwork(**self.conf["model.variance_network"])

        self.renderer = NeuSRenderer(
            self.sdf_network, self.deviation_network, self.color_network,
            **self.conf["model.renderer"],
        )

        params = (
            list(self.sdf_network.parameters())
            + list(self.color_network.parameters())
            + list(self.deviation_network.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.conf["train.learning_rate"])

        self.batch_size = self.conf["train.batch_size"]
        self.end_iter = self.conf["train.end_iter"]
        self.igr_weight = self.conf["train.igr_weight"]
        self.mask_weight = self.conf["train.mask_weight"]
        self.depth_weight = self.conf["train.depth_weight"]

    def get_depth_loss(self, rays_o, rays_d, rendered_depth, img_idx) -> torch.Tensor:
        """
        TODO(you): implement DS-NeRF-style depth supervision.

        Sketch of the approach (see README for the citation trail):
          1. Project self.dataset.sparse_points into image `img_idx` using
             that image's intrinsics/pose to get expected pixel-space depth
             for whichever sparse points land in this ray batch's pixels.
          2. For rays whose pixel matches (or is within ~1px of) a projected
             sparse point, compute a loss between `rendered_depth` (this
             render_core's `depth` output) and the COLMAP point's depth
             along that ray.
          3. DS-NeRF uses a KL-divergence formulation with COLMAP's
             reprojection error as a per-point uncertainty rather than a
             plain L1/L2 -- worth ablating both, plain L1 first.
          4. Weight by `self.depth_weight` (currently 0 in base.conf --
             raise it once this is implemented, and ablate against the
             depth_weight=0 baseline to actually measure whether it helps
             your accuracy metric, per MonoPatchNeRF's finding that this
             kind of supervision helps geometry but not universally).

        Currently returns zero, i.e. this loss is a no-op. This is the
        actual research contribution left for you -- not something to
        hand you pre-solved.
        """
        return torch.tensor(0.0)

    def train_step(self, img_idx: int) -> dict:
        rays_o, rays_d, true_color, mask = self.dataset.gen_random_rays_at(img_idx, self.batch_size)
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

        render_out = self.renderer.render(rays_o, rays_d, near, far)

        color_error = (render_out["color"] - true_color) * mask
        color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / (mask.sum() + 1e-5)

        eikonal_loss = render_out["gradient_error"]

        mask_loss = F.binary_cross_entropy(
            render_out["weights"].sum(dim=1, keepdim=True).clip(1e-3, 1.0 - 1e-3),
            mask, reduction="mean",
        )

        depth_loss = self.get_depth_loss(rays_o, rays_d, render_out["depth"], img_idx)

        loss = (
            color_loss
            + eikonal_loss * self.igr_weight
            + mask_loss * self.mask_weight
            + depth_loss * self.depth_weight
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "color_loss": color_loss.item(),
            "eikonal_loss": eikonal_loss.item(),
            "mask_loss": mask_loss.item(),
            "depth_loss": float(depth_loss) if torch.is_tensor(depth_loss) else depth_loss,
        }

    def train(self):
        for it in tqdm(range(self.end_iter), desc="training"):
            img_idx = np.random.randint(self.dataset.n_images)
            logs = self.train_step(img_idx)

            if it % self.conf["train.report_freq"] == 0:
                tqdm.write(
                    f"iter {it}: loss={logs['loss']:.4f} color={logs['color_loss']:.4f} "
                    f"eikonal={logs['eikonal_loss']:.4f} mask={logs['mask_loss']:.4f} "
                    f"depth={logs['depth_loss']:.4f}"
                )

            if it > 0 and it % self.conf["train.save_freq"] == 0:
                torch.save(
                    {
                        "sdf_network": self.sdf_network.state_dict(),
                        "color_network": self.color_network.state_dict(),
                        "deviation_network": self.deviation_network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "iter": it,
                    },
                    self.exp_dir / f"ckpt_{it:07d}.pth",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=Path, default=Path(__file__).parent / "confs/base.conf")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--exp_dir", type=Path, required=True)
    args = parser.parse_args()

    trainer = SDFTrainer(args.conf, args.data_dir, args.exp_dir)
    trainer.train()
