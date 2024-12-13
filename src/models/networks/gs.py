from typing import *

import kiui
import numpy as np
import torch
from torch import Tensor

from src.options import Options

from .deferred_bp import deferred_bp, render


class GaussianRenderer:
    def __init__(self, opt: Options):

        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        # intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        self.proj_matrix[2, 3] = 1

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def render(
        self,
        gaussians: Tensor,  # (B, N, D)
        cam_view: Tensor,  # (B, V, 4, 4)
        cam_view_proj: Tensor,  # (B, V, 4, 4)
        cam_pos: Tensor,  # (B, V, 3)
        bg_color: Optional[Tensor] = None,
        resolution: Optional[int] = None,
        scale_modifier: float = 1.0,
    ):
        B, V = cam_view.shape[:2]
        S = resolution or self.opt.render_size

        if bg_color is None:
            bg_color = self.bg_color

        if self.opt.render_type == "deferred":
            images, alphas = deferred_bp(
                gaussians,
                S,
                S,
                self.tan_half_fov,
                self.tan_half_fov,
                cam_view,
                cam_pos,
                bg_color,
                scale_modifier,
                self.opt.gs_patch_size,
            )

        else:
            # Batch loop x view loop
            images = []
            alphas = []
            for b in range(B):

                # Gaussian parameters: pos, opacity, scale, rotation, shs
                means3D = gaussians[b, :, 0:3].contiguous().float()
                opacities = gaussians[b, :, 3:4].contiguous().float()
                scales = gaussians[b, :, 4:7].contiguous().float()
                rotations = gaussians[b, :, 7:11].contiguous().float()
                rgbs = gaussians[b, :, 11:].contiguous().float()

                for v in range(V):
                    # Render novel views
                    view_matrix = cam_view[b, v].float()
                    view_proj_matrix = cam_view_proj[b, v].float()
                    campos = cam_pos[b, v].float()

                    render_results = render(
                        # GSRendering settings
                        S,
                        S,
                        self.tan_half_fov,
                        self.tan_half_fov,
                        bg_color,
                        scale_modifier,
                        view_matrix,
                        view_proj_matrix,
                        campos,
                        # GSRendering parameters
                        means3D,
                        rgbs,
                        opacities,
                        scales,
                        rotations,
                    )

                    rendered_image = render_results["image"]
                    rendered_alpha = render_results["alpha"]

                    images.append(rendered_image)
                    alphas.append(rendered_alpha)

            images = torch.stack(images, dim=0).view(B, V, -1, S, S)
            alphas = torch.stack(alphas, dim=0).view(B, V, 1, S, S)

        return {
            "image": images,  # (B, V, C, H, W)
            "alpha": alphas,  # (B, V, 1, H, W)
        }

    def save_ply(self, gaussians: Tensor, path: str, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, "only support batch size 1"

        from plyfile import PlyData, PlyElement

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float()  # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = kiui.op.inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = (
            shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ["x", "y", "z"]
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(scales.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(rotations.shape[1]):
            l.append("rot_{}".format(i))

        dtype_full = [(attribute, "f4") for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")

        PlyData([el]).write(path)

    def load_ply(self, path, compatible=True):

        from plyfile import PlyData

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")
        ]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float()  # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
