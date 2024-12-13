import numpy as np
import torch
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         GaussianRasterizer)
from torch import Tensor
from torch.utils.checkpoint import _get_autocast_kwargs


class DeferredBP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        gaussians,
        height,
        width,
        tan_half_fovx,
        tan_half_fovy,
        cam_view,
        cam_pos,
        bg_color,
        scale_modifier,
        patch_size,
    ):
        assert height % patch_size == 0 and width % patch_size == 0

        ctx.save_for_backward(gaussians)  # save tensors for backward
        ctx.height = height
        ctx.width = width
        ctx.tan_half_fovx = tan_half_fovx
        ctx.tan_half_fovy = tan_half_fovy
        ctx.cam_view = cam_view
        ctx.cam_pos = cam_pos
        ctx.bg_color = bg_color
        ctx.scale_modifier = scale_modifier
        ctx.patch_size = patch_size

        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.manual_seeds = []

        with torch.no_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = cam_view.device
            B, V = cam_view.shape[:2]
            colors = torch.zeros(B, V, 3, height, width, device=device)
            alphas = torch.zeros(B, V, 1, height, width, device=device)

            for b in range(B):
                ctx.manual_seeds.append([])

                # Gaussian parameters: pos, opacity, scale, rotation, shs
                means3D = gaussians[b, :, 0:3].contiguous().float()
                opacities = gaussians[b, :, 3:4].contiguous().float()
                scales = gaussians[b, :, 4:7].contiguous().float()
                rotations = gaussians[b, :, 7:11].contiguous().float()
                rgbs = gaussians[b, :, 11:].contiguous().float()

                for v in range(V):
                    # Render novel views
                    view_matrix = cam_view[b, v].float()
                    campos = cam_pos[b, v].float()

                    for m in range(0, ctx.width // ctx.patch_size):
                        for n in range(0, ctx.height // ctx.patch_size):
                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            fx = 1.0 / (2.0 * tan_half_fovx)
                            fy = 1.0 / (2.0 * tan_half_fovy)
                            cx = 0.5
                            cy = 0.5

                            center_x = (
                                m * ctx.patch_size + ctx.patch_size // 2
                            ) / ctx.width
                            center_y = (
                                n * ctx.patch_size + ctx.patch_size // 2
                            ) / ctx.height

                            scale_x = ctx.width // ctx.patch_size
                            scale_y = ctx.height // ctx.patch_size
                            trans_x = 0.5 - scale_x * center_x
                            trans_y = 0.5 - scale_y * center_y

                            new_fx = scale_x * fx
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y

                            new_tan_half_fovx = 1.0 / (2.0 * new_fx)
                            new_tan_half_fovy = 1.0 / (2.0 * new_fy)
                            new_fovx = 2.0 * np.arctan(new_tan_half_fovx)
                            new_fovy = 2.0 * np.arctan(new_tan_half_fovy)
                            new_shiftx = 2.0 * new_cx - 1.0
                            new_shifty = 2.0 * new_cy - 1.0

                            znear, zfar = 0.01, 1000.0  # TODO: make them configurable
                            proj_matrix = getProjectionMatrix(
                                znear,
                                zfar,
                                new_fovx,
                                new_fovy,
                                new_shiftx,
                                new_shifty,
                                device,
                            )
                            view_proj_matrix = view_matrix @ proj_matrix

                            render_results = render(
                                # GSRendering settings
                                patch_size,
                                patch_size,
                                new_tan_half_fovx,
                                new_tan_half_fovy,
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

                            colors[
                                b,
                                v,
                                :,
                                n * ctx.patch_size : (n + 1) * ctx.patch_size,
                                m * ctx.patch_size : (m + 1) * ctx.patch_size,
                            ] = render_results["image"]
                            alphas[
                                b,
                                v,
                                :,
                                n * ctx.patch_size : (n + 1) * ctx.patch_size,
                                m * ctx.patch_size : (m + 1) * ctx.patch_size,
                            ] = render_results["alpha"]

        return colors, alphas

    @staticmethod
    def backward(ctx, grad_colors, grad_alphas):
        (gaussians,) = ctx.saved_tensors

        gaussians_nosync = gaussians.detach().clone()
        gaussians_nosync.requires_grad = True
        gaussians_nosync.grad = None

        with torch.enable_grad(), torch.cuda.amp.autocast(
            **ctx.gpu_autocast_kwargs
        ), torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
            device = ctx.cam_view.device
            B, V = ctx.cam_view.shape[:2]

            for b in range(B):
                ctx.manual_seeds.append([])

                # Gaussian parameters: pos, opacity, scale, rotation, shs
                means3D = gaussians_nosync[b, :, 0:3].contiguous().float()
                opacities = gaussians_nosync[b, :, 3:4].contiguous().float()
                scales = gaussians_nosync[b, :, 4:7].contiguous().float()
                rotations = gaussians_nosync[b, :, 7:11].contiguous().float()
                rgbs = gaussians_nosync[b, :, 11:].contiguous().float()

                for v in range(V):
                    # Render novel views
                    view_matrix = ctx.cam_view[b, v].float()
                    campos = ctx.cam_pos[b, v].float()

                    for m in range(0, ctx.width // ctx.patch_size):
                        for n in range(0, ctx.height // ctx.patch_size):
                            grad_colors_split = grad_colors[
                                b,
                                v,
                                :,
                                n * ctx.patch_size : (n + 1) * ctx.patch_size,
                                m * ctx.patch_size : (m + 1) * ctx.patch_size,
                            ]
                            grad_alphas_split = grad_alphas[
                                b,
                                v,
                                :,
                                n * ctx.patch_size : (n + 1) * ctx.patch_size,
                                m * ctx.patch_size : (m + 1) * ctx.patch_size,
                            ]

                            seed = torch.randint(0, 2**32, (1,)).long().item()
                            ctx.manual_seeds[-1].append(seed)

                            fx = 1.0 / (2.0 * ctx.tan_half_fovx)
                            fy = 1.0 / (2.0 * ctx.tan_half_fovy)
                            cx = 0.5
                            cy = 0.5

                            center_x = (
                                m * ctx.patch_size + ctx.patch_size // 2
                            ) / ctx.width
                            center_y = (
                                n * ctx.patch_size + ctx.patch_size // 2
                            ) / ctx.height

                            scale_x = ctx.width // ctx.patch_size
                            scale_y = ctx.height // ctx.patch_size
                            trans_x = 0.5 - scale_x * center_x
                            trans_y = 0.5 - scale_y * center_y

                            new_fx = scale_x * fx
                            new_fy = scale_y * fy
                            new_cx = scale_x * cx + trans_x
                            new_cy = scale_y * cy + trans_y

                            new_tan_half_fovx = 1.0 / (2.0 * new_fx)
                            new_tan_half_fovy = 1.0 / (2.0 * new_fy)
                            new_fovx = 2.0 * np.arctan(new_tan_half_fovx)
                            new_fovy = 2.0 * np.arctan(new_tan_half_fovy)
                            new_shiftx = 2.0 * new_cx - 1.0
                            new_shifty = 2.0 * new_cy - 1.0

                            znear, zfar = 0.01, 1000.0  # TODO: make them configurable
                            proj_matrix = getProjectionMatrix(
                                znear,
                                zfar,
                                new_fovx,
                                new_fovy,
                                new_shiftx,
                                new_shifty,
                                device,
                            )
                            view_proj_matrix = view_matrix @ proj_matrix

                            render_results = render(
                                # GSRendering settings
                                ctx.patch_size,
                                ctx.patch_size,
                                new_tan_half_fovx,
                                new_tan_half_fovy,
                                ctx.bg_color,
                                ctx.scale_modifier,
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

                            color_split = render_results["image"]
                            alpha_split = render_results["alpha"]

                            render_split = torch.cat([color_split, alpha_split], dim=0)
                            grad_split = torch.cat(
                                [grad_colors_split, grad_alphas_split], dim=0
                            )
                            render_split.backward(grad_split)

        return (
            gaussians_nosync.grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def deferred_bp(
    gaussians,
    height,
    width,
    tan_half_fovx,
    tan_half_fovy,
    cam_view,
    cam_pos,
    bg_color,
    scale_modifier,
    patch_size,
):
    return DeferredBP.apply(
        gaussians,
        height,
        width,
        tan_half_fovx,
        tan_half_fovy,
        cam_view,
        cam_pos,
        bg_color,
        scale_modifier,
        patch_size,
    )


def render(
    # GSRendering settings
    image_height: int,
    image_width: int,
    tan_half_fovx: float,
    tan_half_fovy: float,
    bg: Tensor,
    scale_modifier: float,
    view_matrix: Tensor,
    view_proj_matrix: Tensor,
    campos: Tensor,
    # GSRendering parameters
    means3D: Tensor,
    colors_precomp: Tensor,
    opacities: Tensor,
    scales: Tensor,
    rotations: Tensor,
):
    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tan_half_fovx,
        tanfovy=tan_half_fovy,
        bg=bg,
        scale_modifier=scale_modifier,
        viewmatrix=view_matrix,
        projmatrix=view_proj_matrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen)
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=torch.zeros_like(means3D),
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    rendered_image = rendered_image.clamp(0, 1)  # RGB

    return {
        "image": rendered_image,
        "alpha": rendered_alpha,
    }


def getProjectionMatrix(znear, zfar, fovX, fovY, shiftX, shiftY, device):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) + shiftX
    P[1, 2] = (top + bottom) / (top - bottom) + shiftY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.transpose(0, 1)
