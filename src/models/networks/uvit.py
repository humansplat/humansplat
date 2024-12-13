from typing import *

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.embeddings import PatchEmbed
from torch import Tensor

from .attention import Attention as CrossAttention
from .embeddings import Timestep, TimestepEmbed

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"


def patchify(images: Tensor, patch_size: int):
    return einops.rearrange(
        images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )


def unpatchify(x: Tensor, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]

    return einops.rearrange(
        x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size
    )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        qk_scale: Optional[float] = None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor):
        B, N, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == "flash":
            qkv = einops.rearrange(
                qkv, "b n (k3 h d) -> k3 b h n d", k3=3, h=self.num_heads
            ).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "b h n d -> b n (h d)")

        elif ATTENTION_MODE == "xformers":
            qkv = einops.rearrange(
                qkv, "b n (k3 h d) -> k3 b n h d", k3=3, h=self.num_heads
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, N, H, D)
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, "b n h d -> b n (h d)")

        elif ATTENTION_MODE == "math":
            qkv = einops.rearrange(
                qkv, "b n (k3 h d) -> k3 b h n d", k3=3, h=self.num_heads
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplementedError

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocalAttention(nn.Module):
    def __init__(self, window_size, embed_dim):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.scale = embed_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x: [batch_size, num_points, embed_dim]
        B, N, D = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, D).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is [B, N, D]

        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create local mask for windowed attention
        idx = torch.arange(N)
        mask_local = abs(idx[:, None] - idx[None, :]) <= self.window_size
        mask_local = mask_local.float()

        # Optional: Apply a global mask, if provided
        if mask is not None:
            mask_local = mask_local * mask

        # Mask out values beyond the attention window
        attn = attn.masked_fill(mask_local == 0, float("-inf"))

        # Softmax to obtain attention probabilities
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Project the output
        return self.proj(out)


class MVAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        qk_scale: Optional[float] = None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_frames=4,
    ):
        super().__init__()

        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_frames = num_frames

    def forward(self, x: Tensor):
        x = einops.rearrange(x, "(b v) n d -> b (v n) d", v=self.num_frames)
        x = self.attn(x)
        x = einops.rearrange(x, "b (v n) d -> (b v) n d", v=self.num_frames)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale: Optional[float] = None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
        use_checkpoint=False,
        use_mvattn=False,
        num_frames=4,
        use_cross_attn=False,
        dim_kv: Optional[int] = None,
        k_window_size: Optional[int] = None,
    ):
        super().__init__()

        if use_cross_attn:
            self.norm0 = norm_layer(dim)
            self.cross_attn = CrossAttention(
                dim,
                dim_kv=dim_kv,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )

        self.norm1 = norm_layer(dim)
        if use_mvattn:
            self.attn = MVAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                num_frames=num_frames,
            )
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale
            )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames

        self.k_window_size = k_window_size
        self.N = 64
        self.grids = self.generate_grid(self.N)  # shape of VAE feature

    def forward(self, x, skip=None, context=None, proj_xy=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, skip, context, proj_xy
            )
        else:
            return self._forward(x, skip, context, proj_xy)

    def generate_grid(self, N):
        grid = torch.stack(
            torch.meshgrid(torch.arange(N), torch.arange(N)), -1
        ).reshape(-1, 2)
        return grid

    # project aware attention
    def points_in_window(self, centers, K):
        batch_size = centers.size(0)
        expanded_grid = (
            self.grids.unsqueeze(0).repeat(batch_size, 1, 1).to(centers.device)
        )
        expanded_centers = (
            centers.unsqueeze(1).repeat(1, self.N * self.N, 1).to(centers.device)
        )
        min_coords = expanded_centers / 8 - K  # [batch_size, N*N, 2]
        max_coords = expanded_centers / 8 + K  # [batch_size, N*N, 2]
        mask = (expanded_grid >= min_coords) & (
            expanded_grid <= max_coords
        )  # [batch_size, N*N, 2]
        mask = mask.all(dim=2)  # [batch_size, N*N]
        return mask

    def _forward(self, x: Tensor, skip=None, context=None, proj_xy=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        if context is not None:
            # x: [B, 64x64+1, C], context: [B, N_p, C]
            # project aware attention with window size
            context = einops.repeat(context, "b n d -> (b v) n d", v=self.num_frames)
            x = x + self.cross_attn(self.norm0(x), context)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=14,
        img_size=256,
        patch_size=16,
        splat_size=64,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale: Optional[float] = None,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        conv=True,
        skip=True,
        use_mvattn=False,
        num_frames=4,
        use_t_cond=False,
        use_cross_attn=False,
        dim_kv: Optional[int] = None,
        k_window_size: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv = conv
        self.use_t_cond = use_t_cond
        self.patch_embed = PatchEmbed(
            img_size, img_size, patch_size, in_chans, embed_dim
        )  # include 2d pos emb
        if use_t_cond:
            self.time_embed = nn.Sequential(
                Timestep(embed_dim), TimestepEmbed(embed_dim, embed_dim)
            )
        self.in_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    use_mvattn=use_mvattn,
                    num_frames=num_frames,
                    use_cross_attn=use_cross_attn,
                    dim_kv=dim_kv,
                    k_window_size=k_window_size,
                )
                for _ in range(depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            use_mvattn=use_mvattn,
            num_frames=num_frames,
            use_cross_attn=use_cross_attn,
            dim_kv=dim_kv,
            k_window_size=k_window_size,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    skip=skip,
                    use_checkpoint=use_checkpoint,
                    use_mvattn=use_mvattn,
                    num_frames=num_frames,
                    use_cross_attn=use_cross_attn,
                    dim_kv=dim_kv,
                    k_window_size=k_window_size,
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(
            embed_dim,
            min(2, int(splat_size / (img_size / patch_size))) ** 2
            * (embed_dim if conv else out_chans),
        )
        self.final_layer = (
            nn.Sequential(
                nn.Conv2d(embed_dim, out_chans, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(out_chans, out_chans, 3, padding=1),
            )
            if conv
            else nn.Identity()
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.splat_size = splat_size

    def forward(
        self,
        x: Tensor,
        t: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        proj_xy: Optional[Tensor] = None,
    ):
        if t is not None:
            if not torch.is_tensor(t):
                if isinstance(t, (int, float)):  # single timestep
                    t = torch.tensor([t], device=x.device)
                else:  # list of timesteps
                    assert len(t) == x.shape[0]
                    t = torch.tensor(t, device=x.device)
            else:  # is tensor
                if t.dim() == 0:
                    t = t.unsqueeze(-1).to(x.device)
                else:
                    assert t.dim() == 1
            # Broadcast to batch dimension, in a way that's campatible with ONNX/Core ML
            t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)  # (B*V,)

        x = self.patch_embed(x)  # (B, N, D); include 2d pos emb

        if self.use_t_cond:
            time_emb = self.time_embed(t)  # (B, D)
            x = torch.cat([time_emb.unsqueeze(1), x], dim=1)  # (B, 1+N, D)

        skips = []
        for blk in self.in_blocks:
            x = blk(x, None, context, proj_xy)
            skips.append(x)

        x = self.mid_block(x, None, context, proj_xy)

        for blk in self.out_blocks:
            x = blk(x, skips.pop(), context, proj_xy)

        if self.use_t_cond:
            x = x[:, 1:, :]

        x = self.norm(x)
        x = self.decoder_pred(x)

        x = unpatchify(x, self.embed_dim if self.conv else self.out_chans)
        if int(self.splat_size / (self.img_size / self.patch_size)) > 2:
            scale_factor = int(self.splat_size / (self.img_size / self.patch_size)) // 2
            x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        x = self.final_layer(x)

        return x
