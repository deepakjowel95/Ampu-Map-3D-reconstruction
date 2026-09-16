"""
Network definitions for SDF-based surface reconstruction (NeuS/VolSDF-style).

Three networks, matched to the NeuS architecture (Wang et al. 2021):
  - SDFNetwork: maps a 3D point -> (signed distance, 256-d feature vector).
    Geometric initialization biases it to start as roughly a sphere, which
    stabilizes early training -- without this, SDF training routinely
    diverges before it learns anything useful.
  - RenderingNetwork: maps (point, normal, view direction, SDF feature) -> RGB.
    View-dependence lets it model skin's actual specular/soft-shading
    response instead of assuming pure Lambertian reflectance.
  - SingleVarianceNetwork: the single learnable scalar 's' controlling how
    sharply the SDF converts to a rendering density (NeuS Eq. 3) --
    starts blurry and sharpens automatically as training progresses.
"""
import numpy as np
import torch
import torch.nn as nn


class Embedder:
    """Positional encoding (NeRF-style Fourier features)."""

    def __init__(self, input_dims: int, multires: int, include_input: bool = True):
        self.include_input = include_input
        self.input_dims = input_dims
        freq_bands = 2.0 ** torch.linspace(0.0, multires - 1, multires)
        self.freq_bands = freq_bands
        self.out_dim = input_dims * (include_input + 2 * multires)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x] if self.include_input else []
        for freq in self.freq_bands:
            outs.append(torch.sin(x * freq))
            outs.append(torch.cos(x * freq))
        return torch.cat(outs, dim=-1)


class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_out=257, d_hidden=256, n_layers=8,
                 multires=6, bias=0.5, scale=1.0, geometric_init=True,
                 weight_norm=True, skip_in=(4,)):
        super().__init__()
        self.embedder = Embedder(d_in, multires) if multires > 0 else None
        in_dim = self.embedder.out_dim if self.embedder else d_in
        self.skip_in = skip_in
        self.scale = scale

        dims = [in_dim] + [d_hidden] * n_layers + [d_out]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            out_dim = dims[l + 1]
            if l + 1 in skip_in:
                out_dim -= in_dim
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                # Sphere initialization (Atzmon & Lipman 2020 / NeuS): biases
                # the network to represent an approximate sphere at init,
                # so the eikonal loss has a sane gradient from step 0 instead
                # of training against an arbitrary random SDF.
                if l == self.num_layers - 2:
                    nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=1e-4)
                    nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.constant_(lin.weight[:, 3:], 0.0)
                    nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in skip_in:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.weight[:, -(in_dim - 3):], 0.0)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, f"lin{l}", lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x * self.scale
        feat = self.embedder.embed(x_in) if self.embedder else x_in
        h = feat
        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin{l}")
            if l in self.skip_in:
                h = torch.cat([h, feat], dim=-1) / np.sqrt(2)
            h = lin(h)
            if l < self.num_layers - 2:
                h = self.activation(h)
        # h[:, :1] is signed distance (rescaled back to true units), rest is feature
        return torch.cat([h[:, :1] / self.scale, h[:, 1:]], dim=-1)

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, :1]

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """d(SDF)/dx via autograd -- used both as the surface normal and in
        the eikonal regularization loss (|grad| should be ~1 everywhere)."""
        x.requires_grad_(True)
        y = self.sdf(x)
        grad = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        return grad


class RenderingNetwork(nn.Module):
    def __init__(self, d_feature=256, d_in=9, d_out=3, d_hidden=256,
                 n_layers=4, multires_view=4, weight_norm=True):
        super().__init__()
        self.embedder = Embedder(3, multires_view) if multires_view > 0 else None
        view_dim = self.embedder.out_dim if self.embedder else 3
        # d_in accounts for: point(3) + normal(3) + view_dir(3) at raw dims;
        # actual input dim is recomputed below once view encoding is known.
        in_dim = 3 + 3 + view_dim + d_feature
        dims = [in_dim] + [d_hidden] * n_layers + [d_out]
        self.num_layers = len(dims)
        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            if weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, f"lin{l}", lin)
        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        view_enc = self.embedder.embed(view_dirs) if self.embedder else view_dirs
        h = torch.cat([points, view_enc, normals, feature_vectors], dim=-1)
        for l in range(self.num_layers - 1):
            h = getattr(self, f"lin{l}")(h)
            if l < self.num_layers - 2:
                h = self.relu(h)
        return torch.sigmoid(h)


class SingleVarianceNetwork(nn.Module):
    """The scalar 's' in NeuS Eq. 3, learned in log-space, shared across all rays."""

    def __init__(self, init_val=0.3):
        super().__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, batch_size: int) -> torch.Tensor:
        return torch.ones(batch_size, 1) * torch.exp(self.variance * 10.0)
