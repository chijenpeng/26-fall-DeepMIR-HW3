"""Adaptive Projected Guidance (APG) for the v-prediction Stable Audio Open sampler.

Sadat, Hilliges, Weber, "Eliminating Oversaturation and Artifacts of High Guidance Scales in
Diffusion Models", ICLR 2025 (arXiv 2410.02416), Algorithm 1. Reference implementation:
diffusers 0.41 `diffusers/guiders/adaptive_projected_guidance.py` (`normalized_guidance`,
`MomentumBuffer`); order of operations there and here: momentum -> rescale -> projection.

APG works on the *denoised (x0)* prediction D. With  delta = D(c) - D(null):
    delta <- delta + beta * running_delta            (reverse momentum, beta < 0)
    delta <- delta * min(1, r / ||delta||)           (norm rescaling, norm over all non-batch dims)
    delta_par  = projection of delta onto D(c)       (all non-batch dims)
    delta_orth = delta - delta_par
    D_apg = D(c) + (w - 1) * (delta_orth + eta * delta_par)
eta = 1, r = inf, beta = 0 gives plain CFG.

The cosine-DPM-solver scheduler of Stable Audio Open is EDM-preconditioned and linear in the model
output v:  D = c_skip * x + c_out * v,  c_skip = 1/(sigma^2+1),  c_out = -sigma/sqrt(sigma^2+1)
(sigma_data = 1). `apg_v_prediction` converts v -> D, applies APG, and converts back to v.
"""

import torch


class MomentumBuffer:
    """running = update + momentum * running (same as diffusers' MomentumBuffer)."""

    def __init__(self, momentum):
        self.momentum = momentum
        self.running = 0

    def update(self, x):
        self.running = x + self.momentum * self.running
        return self.running


def apg(d_cond, d_uncond, scale, eta=0.0, norm_threshold=0.0, buffer=None, trace=None):
    """APG on denoised predictions (any shape (B, ...)). norm_threshold <= 0 disables rescaling."""
    dims = list(range(1, d_cond.ndim))
    delta = d_cond - d_uncond
    if trace is not None:
        trace["delta_norm"] = delta.flatten(1).norm(dim=1).item()   # raw ||D(c) - D(null)||
    if buffer is not None:
        delta = buffer.update(delta)
    if norm_threshold > 0:
        norm = delta.norm(dim=dims, keepdim=True)
        delta = delta * torch.clamp(norm_threshold / norm, max=1.0)
    unit = torch.nn.functional.normalize(d_cond, dim=dims)
    par = (delta * unit).sum(dim=dims, keepdim=True) * unit
    orth = delta - par
    return d_cond + (scale - 1.0) * (orth + eta * par)


def apg_v_prediction(sigma, x, v_uncond, v_cond, scale, eta=0.0, norm_threshold=0.0, buffer=None, trace=None):
    """Apply APG in denoised space to v-prediction outputs; returns the guided v (input dtype).

    sigma: float, current noise level scheduler.sigmas[i]; x: current noisy latent.
    """
    dtype = v_cond.dtype
    x, v_u, v_c = x.float(), v_uncond.float(), v_cond.float()
    c_skip = 1.0 / (sigma**2 + 1.0)
    c_out = -sigma / (sigma**2 + 1.0) ** 0.5
    d_u, d_c = c_skip * x + c_out * v_u, c_skip * x + c_out * v_c
    d_g = apg(d_c, d_u, scale, eta, norm_threshold, buffer, trace)
    return ((d_g - c_skip * x) / c_out).to(dtype)
