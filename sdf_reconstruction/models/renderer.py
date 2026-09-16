"""
Volume renderer that converts an SDF into per-sample rendering weights
(NeuS Wang et al. 2021, Eq. 13) instead of the density-field weights a
vanilla NeRF would use. This is the actual mechanism behind "SDF-based
surface representation improves geometric accuracy over density-field NeRF":
weights are derived from the SDF's sign change (i.e. an explicit surface
crossing) rather than from an unconstrained density MLP, which is what
gives NeuS its stronger surface inductive bias.

Also returns per-ray expected depth (`mid_z_vals` weighted by rendering
weights) -- this is the quantity `exp_runner.get_depth_loss()` should
supervise against COLMAP sparse depth.
"""
import torch
import torch.nn.functional as F


class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network, color_network,
                 n_samples=64, n_importance=64, up_sample_steps=4,
                 perturb=1.0):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core(self, rays_o, rays_d, z_vals, sdf_network, deviation_network,
                     color_network, background_rgb=None):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts)
        normals = F.normalize(gradients, dim=-1)

        sampled_color = color_network(pts, normals, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(1).clamp(1e-6, 1e6)  # single learnable scalar, Eq. 3
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        # "Cosine annealing" (NeuS): only start using the true ray-surface
        # angle once normals are meaningful; before that fall back to
        # treating the ray as hitting head-on. Prevents early-training
        # instability from noisy gradients.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * 1.0 + F.relu(-true_cos) * 0.0)

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        transmittance = torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1.0 - alpha + 1e-7], dim=-1), dim=-1
        )[:, :-1]
        weights = alpha * transmittance

        color = (sampled_color.reshape(batch_size, n_samples, 3) * weights[..., None]).sum(dim=1)
        depth = (mid_z_vals * weights).sum(dim=1, keepdim=True)  # expected termination distance
        acc = weights.sum(dim=1, keepdim=True)

        if background_rgb is not None:
            color = color + background_rgb * (1.0 - acc)

        # Eikonal term: |grad SDF| should be 1 everywhere (Gropp et al. 2020)
        gradient_error = ((torch.linalg.norm(gradients, ord=2, dim=-1) - 1.0) ** 2).mean()

        return {
            "color": color,
            "depth": depth,
            "weights": weights,
            "gradient_error": gradient_error,
            "sdf": sdf.reshape(batch_size, n_samples),
            "gradients": gradients.reshape(batch_size, n_samples, 3),
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """Hierarchical importance sampling: concentrate more samples near
        the current SDF's zero-crossing estimate, same role as NeRF's
        coarse->fine sampling but driven by the SDF sign change."""
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1)
        inside_sphere = radius[:, :-1] < 1.0

        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z, next_z = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z - prev_z + 1e-5)
        cos_val = torch.clamp(cos_val, max=0.0)  # only care about the ray entering the surface

        dist = next_z - prev_z
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)
        alpha = alpha * inside_sphere.float()

        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1.0 - alpha + 1e-7], dim=-1), dim=-1
        )[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True)
        return z_samples

    def render(self, rays_o, rays_d, near, far, background_rgb=None):
        batch_size = rays_o.shape[0]
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)[None, :].expand(batch_size, -1)
        z_vals = near + (far - near) * z_vals

        if self.perturb > 0:
            t_rand = (torch.rand_like(z_vals) - 0.5)
            z_vals = z_vals + t_rand * (far - near) / self.n_samples

        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    n_new = self.n_importance // self.up_sample_steps
                    inv_s = 64 * 2 ** i
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, n_new, inv_s)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == self.up_sample_steps))

        return self.render_core(rays_o, rays_d, z_vals, self.sdf_network,
                                 self.deviation_network, self.color_network,
                                 background_rgb=background_rgb)

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        sdf = torch.gather(sdf, 1, index)
        return z_vals, sdf


def sample_pdf(bins, weights, n_samples, det=False):
    """Inverse-CDF sampling given a piecewise-constant weight distribution."""
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    if det:
        u = torch.linspace(0.0, 1.0, n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).contiguous()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples
