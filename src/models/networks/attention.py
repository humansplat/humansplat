from typing import *

import einops
import torch
from torch import Tensor, nn

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: Optional[int] = None,
        dim_kv: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        dim_q = dim_q if dim_q is not None else dim
        dim_kv = dim_kv if dim_kv is not None else dim_q

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_kv, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_kv, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, kv: Optional[Tensor] = None) -> Tensor:
        if kv is None:
            kv = q  # i.e., self attention

        q, k, v = self.to_q(q), self.to_k(kv), self.to_v(kv)

        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = einops.rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        if ATTENTION_MODE == "math":
            attn = self.scale * q @ k.transpose(-2, -1)  # (B, H, N, M)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = einops.rearrange(attn @ v, "b h n d -> b n (h d)")
        elif ATTENTION_MODE == "flash":
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "b h n d -> b n (h d)")
        elif ATTENTION_MODE == "xformers":
            q = einops.rearrange(q, "b h n d -> b n h d")
            k = einops.rearrange(k, "b h n d -> b n h d")
            v = einops.rearrange(v, "b h n d -> b n h d")
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "b n h d -> b n (h d)")
        else:
            raise ValueError(f"Invalid attention mode: [{ATTENTION_MODE}]")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
