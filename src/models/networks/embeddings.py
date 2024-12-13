import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_1d_sincos_encode(
    steps: Tensor, emb_dim: int, max_period: int = 10000
) -> Tensor:
    """Get sinusoidal encodings for a batch of timesteps/positions."""
    assert (
        steps.dim() == 1
    ), f"Parameter `steps` must be a 1D tensor, but got {steps.dim()}D."

    half_dim = emb_dim // 2
    emb = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, device=steps.device).float()
        / half_dim
    )
    emb = steps[:, None].float() * emb[None, :]  # (num_steps, half_dim)

    # Concat sine and cosine encodings
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # (num_steps, emb_dim)

    # Zero padding
    if emb_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))

    assert emb.shape == (steps.shape[0], emb_dim)
    return emb


class Timestep(nn.Module):
    """Encode timesteps with sinusoidal encodings."""

    def __init__(self, time_emb_dim: int):
        super().__init__()

        self.time_emb_dim = time_emb_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        return get_1d_sincos_encode(timesteps, self.time_emb_dim)


class TimestepEmbed(nn.Module):
    """Embed sinusoidal encodings with a 2-layer MLP."""

    def __init__(self, in_dim: int, time_emb_dim: int, act_fn_name: str = "SiLU"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, time_emb_dim),
            getattr(nn, act_fn_name)(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, sample: Tensor) -> Tensor:
        return self.mlp(sample)


class SMPLEmbed(nn.Module):
    """Embed sinusoidal encodings with a 2-layer MLP."""

    def __init__(self, in_dim: int, time_emb_dim: int, act_fn_name: str = "SiLU"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, time_emb_dim),
            getattr(nn, act_fn_name)(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, sample: Tensor) -> Tensor:
        return self.mlp(sample)


class AdaGroupNorm(nn.Module):
    def __init__(self, t_dim: int, out_dim: int, num_groups: int, eps: float = 1e-5):
        super().__init__()

        self.num_groups = num_groups
        self.eps = eps

        self.emb = nn.Sequential(Timestep(t_dim), TimestepEmbed(t_dim, out_dim * 2))

    def forward(self, x: Tensor, t: torch.LongTensor):
        emb = self.emb(t)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1.0 + scale) + shift
        return x
