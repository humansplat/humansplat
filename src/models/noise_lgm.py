import os
from typing import *

import einops
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from src.options import Options

from .human_rec import HumanRec
from .sv3d_p import SV3D

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class NoiseLGM(nn.Module):
    def __init__(self, opt: Options, weight_dtype=torch.bfloat16):
        super().__init__()

        self.opt = opt
        self.sv3d = SV3D(opt, weight_dtype)
        self.sv3d.requires_grad_(False)

        self.humanrec = HumanRec(opt, weight_dtype)
        delattr(self.humanrec, "vae")  # not used; remove to save memory
        self.iter_step = 0

        infer_from_iter = 178000
        ckpt_dir = "out"
        from safetensors.torch import load_model as safetensors_load_model

        safetensors_load_model(
            self.sv3d,
            os.path.join(ckpt_dir, f"{infer_from_iter:06d}", "model.safetensors"),
            strict=False,
        )
        logger.info(f"Load checkpoint from {ckpt_dir}  iteration [{infer_from_iter}]\n")

    def state_dict(self, **kwargs):
        # Remove frozen pretrained parameters
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k or "sv3d" in k:
                del state_dict[k]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):
        self.sv3d.eval()

        self.iter_step = self.iter_step + 1

        outputs = {}
        images = data["images_output"].to(dtype=weight_dtype)  # (B, V, 3, H, W); [0, 1]
        cam_poses = data["cam_poses_output"].to(dtype=weight_dtype)  # (B, V, 4)
        V, device = images.shape[1], images.device

        # SV3D
        with torch.no_grad():
            pred_original_latents, _ = self.sv3d.diffuse_and_denoise(
                images, cam_poses, noises=None
            )

        # humanrec
        cam_poses_4x4 = data["cam_poses_4x4_output"].to(
            dtype=weight_dtype
        )  # (B, V, 4, 4)

        # Randomly choose 7 views; one fixed view for the input conditional image
        random_idxs = torch.randint(1, V, size=(7,))  # TODO: make `7` configurable
        random_idxs = torch.concat(
            [torch.zeros_like(random_idxs)[0:1], random_idxs]
        ).long()

        # `4` is hard-coded for humanrec
        latents = pred_original_latents[:, random_idxs[:4], ...]
        cam_poses_4x4 = cam_poses_4x4[:, random_idxs[:4], ...]

        render_outputs, _ = self.humanrec.reconstruct(
            latents,
            cam_poses_4x4,
            data["cam_view"][:, random_idxs, ...],
            data["cam_view_proj"][:, random_idxs, ...],
            data["cam_pos"][:, random_idxs, ...],
            t=None,
            input_latents=True,
            input_data=data,
        )

        pred_images = render_outputs["image"]  # (B, V', C, H, W)
        pred_alphas = render_outputs["alpha"]  # (B, V', 1, H, W)
        gt_images = data["images_output"][:, random_idxs, ...]  # (B, V', 3, H, W)
        gt_masks = data["masks_output"][:, random_idxs, ...]  # (B, V', 1, H, W)
        # Denoised images by SV3D, for visualization
        pred_original_latents_subset = einops.rearrange(
            pred_original_latents[:, random_idxs, ...], "b v c h w -> (b v) c h w"
        )
        denoised_images = self.sv3d.vae.decode(
            1.0 / self.sv3d.vae.config.scaling_factor * pred_original_latents_subset
        ).sample
        denoised_images = einops.rearrange(
            denoised_images, "(b v) c h w -> b v c h w", v=len(random_idxs)
        )
        denoised_images = (denoised_images * 0.5 + 0.5).clamp(0.0, 1.0)

        # For visualization
        outputs["images_pred"] = pred_images
        outputs["images_denoised"] = denoised_images
        outputs["images_gt"] = gt_images

        outputs["image_mse"] = image_mse = F.mse_loss(pred_images, gt_images)
        outputs["mask_mse"] = mask_mse = F.mse_loss(pred_alphas, gt_masks)
        loss = image_mse + mask_mse

        if self.opt.lambda_lpips > 0.0:
            lpips = self.humanrec.lpips_loss(
                # Downsampled to at most 256 to reduce memory cost
                F.interpolate(
                    einops.rearrange(gt_images, "b v c h w -> (b v) c h w") * 2.0 - 1.0,
                    (256, 256),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    einops.rearrange(pred_images, "b v c h w -> (b v) c h w") * 2.0
                    - 1.0,
                    (256, 256),
                    mode="bilinear",
                    align_corners=False,
                ),
            ).mean()
            outputs["lpips"] = lpips
            loss += self.opt.lambda_lpips * lpips
        else:
            with torch.no_grad():
                lpips = self.humanrec.lpips_loss(
                    # Downsampled to at most 256 to reduce memory cost
                    F.interpolate(
                        einops.rearrange(gt_images, "b v c h w -> (b v) c h w") * 2.0
                        - 1.0,
                        (256, 256),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    F.interpolate(
                        einops.rearrange(pred_images, "b v c h w -> (b v) c h w") * 2.0
                        - 1.0,
                        (256, 256),
                        mode="bilinear",
                        align_corners=False,
                    ),
                ).mean()
                outputs["lpips"] = lpips

        outputs["loss"] = loss

        # Metric: PNSR
        with torch.no_grad():
            psnr = -10 * torch.log10(
                torch.mean((pred_images.detach() - gt_images) ** 2)
            )
            outputs["psnr"] = psnr

        return outputs
