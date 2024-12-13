from typing import *

import torch
import torch.nn.functional as F
from kiui.lpips import LPIPS
from torch import Tensor, nn

from src.options import Options


class HumanLoss(nn.Module):
    def __init__(self, opt: Options) -> None:
        super().__init__(opt)
        self.mse = nn.MSELoss()
        self.lambda_mask = 1
        self.lambda_mse = 1
        self.lambda_lpips = 1
        self.lambda_hierarchical = 1
        self.lambda_roi_joint_region = 2

        self.lpips_loss = LPIPS(net="vgg")
        self.lpips_loss.requires_grad_(False)
        self.n_levels = 3
        self.opt = opt

    def forward(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):
        outputs = {}
        device = self.vae.device
        images = (
            data["images_input"].to(dtype=weight_dtype).to(device)
        )  # (B, V_in, 3, H, W)
        cam_poses_4x4_input = (
            data["cam_poses_4x4_input"].to(dtype=weight_dtype).to(device)
        )  # (B, V_in, 4, 4)

        if self.opt.load_smpl:
            smpl_data = data["smpl_xyz_rgb"].to(device)  # (B,); list of stirngs
        else:
            smpl_data = None

        render_outputs, _ = self.reconstruct(
            images,
            cam_poses_4x4_input,
            data["cam_view"].to(device),
            data["cam_view_proj"].to(device),
            data["cam_pos"].to(device),
            captions=smpl_data,
        )

        pred_images = render_outputs["image"].to(device)  # (B, V, C, H, W)
        pred_alphas = render_outputs["alpha"].to(device)  # (B, V, 1, H, W)

        gt_images = data["images_output"].to(device)  # (B, V, 3, H, W)
        if self.opt.load_normal:
            gt_images = gt_images[:, :, :3, ...]  # only use rgb for supervision
        gt_masks = data["masks_output"].to(device)  # (B, V, 1, H, W)

        # For visualization
        outputs["images_pred"] = pred_images
        outputs["images_gt"] = gt_images

        outputs["image_mse"] = image_mse = F.mse_loss(pred_images, gt_images)
        outputs["mask_mse"] = mask_mse = F.mse_loss(pred_alphas, gt_masks)
        loss = image_mse + mask_mse

        if self.opt.lambda_lpips > 0.0:
            lpips = self.lpips_loss(
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
                lpips = self.lpips_loss(
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

        return outputs
