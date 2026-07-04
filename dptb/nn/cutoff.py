import math
import torch


@torch.jit.script
def cosine_cutoff(x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8):
    """A piecewise cosine cutoff starting the cosine decay at r_decay_factor*r_max.

    Broadcasts over r_max.
    """
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    r_decay: torch.Tensor = r_start_cos_ratio * r_max
    # for x < r_decay, clamps to 1, for x > r_max, clamps to 0
    x = x.clamp(r_decay, r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


@torch.jit.script
def polynomial_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    """Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


    Parameters
    ----------
    r_max : tensor
        Broadcasts over r_max.

    p : int
        Power used in envelope function
    """
    assert p >= 2.0
    r_max, x = torch.broadcast_tensors(r_max.unsqueeze(-1), x.unsqueeze(0))
    x = x / r_max

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)

@torch.jit.script
def polynomial_cutoff2(
    r: torch.Tensor, rc: torch.Tensor, rs: torch.Tensor,
) -> torch.Tensor:
    
    r_ = torch.zeros_like(r)
    r_[r<rs] = 1/r[r<rs]
    x = (r - rc) / (rs - rc)
    mid_mask = (rs<=r) * (r < rc)
    r_[mid_mask] = 1/r[mid_mask] * (x[mid_mask]**3 * (10 + x[mid_mask] * (-15 + 6 * x[mid_mask])) + 1)

    return r_

def boundary_envelope(r: torch.Tensor, r_max, onset: float = 0.95) -> torch.Tensor:
    """C2-smooth switching envelope: exactly 1 for r <= onset*r_max, smootherstep down to 0 at r_max.

    Used to make cutoff-boundary behaviour continuous: several pathways (message-passing features /
    additive shifts) are otherwise O(1) discontinuous at the cutoff-activation boundary, and the p=6
    polynomial cutoff evaluated at r/r_max ~ 1 suffers catastrophic cancellation in float32
    (absolute noise ~3e-6 vs true values 1e-6..1e-11), so active-set membership can flip between
    dtypes/platforms and boundary edges pick up O(0.1-1 eV) prediction noise.

    Numerical stability at t -> 1 comes from the exact factorization
        1 - (10 t^3 - 15 t^4 + 6 t^5) = (1-t)^3 (6 t^2 + 3 t + 1),
    which is cancellation-free.
    """
    t = ((r - onset * r_max) / ((1.0 - onset) * r_max)).clamp(0.0, 1.0)
    omt = 1.0 - t
    return omt * omt * omt * (6.0 * t * t + 3.0 * t + 1.0)
