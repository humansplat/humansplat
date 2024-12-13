from typing import *

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import Attention


class MVAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: Optional[int] = None,
        dim_kv: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1.0,
        num_frames: int = 4,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(
            num_groups=groups, num_channels=dim, eps=eps, affine=True
        )
        self.attn = Attention(
            dim, dim_q, dim_kv, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop
        )

    def forward(self, x: Tensor):
        res = x
        x = self.norm(x)

        x = einops.rearrange(x, "(b v) d h w -> b (v h w) d", v=self.num_frames)
        x = self.attn(x)
        x = einops.rearrange(
            x, "b (v h w) d -> (b v) d h w", v=self.num_frames, w=res.shape[-1]
        )

        if self.residual:
            x = (x + res) * self.skip_scale
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1.0,  # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(groups, in_channels, eps, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.norm2 = nn.GroupNorm(groups, out_channels, eps, affine=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        self.act = F.silu  # TODO: make it configurable

        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor):
        res = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale
        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1.0,
        num_frames: int = 4,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(
                        out_channels,
                        None,
                        None,
                        attention_heads,
                        skip_scale=skip_scale,
                        num_frames=num_frames,
                    )
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x: Tensor):
        xs = []

        for attn, net in zip(self.attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1.0,
        num_frames: int = 4,
    ):
        super().__init__()

        nets = []
        attns = []
        # First layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # More layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(
                        in_channels,
                        None,
                        None,
                        attention_heads,
                        skip_scale=skip_scale,
                        num_frames=num_frames,
                    )
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

    def forward(self, x: Tensor):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1.0,
        num_frames: int = 4,
    ):
        super().__init__()

        nets = []
        attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            if prev_out_channels == 0:  # no skip-connection
                cskip = 0
            else:
                cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(
                    MVAttention(
                        out_channels,
                        None,
                        None,
                        attention_heads,
                        skip_scale=skip_scale,
                        num_frames=num_frames,
                    )
                )
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: Tensor, xs: Optional[List[Tensor]] = None):
        for attn, net in zip(self.attns, self.nets):
            if xs is not None:  # for asymetric skip-connection
                res_x = xs[-1]
                xs = xs[:-1]
                x = torch.cat([x, res_x], dim=1)
            x = net(x)
            if attn:
                x = attn(x)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = self.upsample(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256),
        up_attention: Tuple[bool, ...] = (True, True, False),
        layers_per_block: int = 2,
        skip_scale: float = (0.5**0.5),
        num_frames: int = 4,
    ):
        super().__init__()

        # First layer
        self.conv_in = nn.Conv2d(
            in_channels, down_channels[0], kernel_size=3, stride=1, padding=1
        )

        # Down blocks
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(
                DownBlock(
                    cin,
                    cout,
                    num_layers=layers_per_block,
                    downsample=(i != len(down_channels) - 1),  # not final layer
                    attention=down_attention[i],
                    skip_scale=skip_scale,
                    num_frames=num_frames,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # Mid block
        self.mid_block = MidBlock(
            down_channels[-1],
            attention=mid_attention,
            skip_scale=skip_scale,
            num_frames=num_frames,
        )

        # Up blocks
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            # For asymetric
            if i < len(down_channels):
                cskip = down_channels[max(-2 - i, -len(down_channels))]
            else:
                cskip = 0  # no skip-connection

            up_blocks.append(
                UpBlock(
                    cin,
                    cskip,
                    cout,
                    num_layers=layers_per_block + 1,  # one more layer for up
                    upsample=(i != len(up_channels) - 1),  # not final layer
                    attention=up_attention[i],
                    skip_scale=skip_scale,
                    num_frames=num_frames,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # Last layer
        self.norm_out = nn.GroupNorm(
            num_channels=up_channels[-1], num_groups=32, eps=1e-5
        )
        self.conv_out = nn.Conv2d(
            up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor):
        # First layer
        x = self.conv_in(x)

        # Down blocks
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss.extend(xs)

        # Mid block
        x = self.mid_block(x)

        # Up blocks
        for block in self.up_blocks:
            if xss is None or len(xss) < len(block.nets):  # no skip-connection
                xs, xss = None, None
            else:
                xs = xss[-len(block.nets) :]
                xss = xss[: -len(block.nets)]
            x = block(x, xs)

        # Last layer
        x = self.norm_out(x)
        x = F.silu(x)  # TODO: make it configurable
        x = self.conv_out(x)
        return x
