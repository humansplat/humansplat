from copy import deepcopy
from typing import *

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as calculate_ssim
from torch import Tensor, nn
from tqdm import tqdm

from src.models.sv3d.pipeline_sv3d import _resize_with_antialiasing
from src.options import Options

from .lgm import LGM
from .sv3d_p import SV3D

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class SV3DLGM3(nn.Module):
    def __init__(self, opt: Options, weight_dtype=torch.bfloat16):
        super().__init__()

        self.opt = opt

        self.sv3d = SV3D(opt, weight_dtype)
        self.lgm = LGM(opt, weight_dtype, use_t_cond=True)
        delattr(self.lgm, "vae")  # not used; to save memory

        # TODO: Customize the parameters that need to be trained
        for name, param in self.sv3d.unet.named_parameters():
            if "transformer_blocks" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def state_dict(self, **kwargs):
        # Remove frozen pretrained parameters
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if (
                "lpips_loss" in k
                or ("sv3d" in k and "unet" not in k)
                or ("sv3d" in k and "unet" in k and "transformer_blocks" not in k)
            ):
                del state_dict[k]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(
        self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16, vis=False
    ):

        outputs = {}

        images = data["images_output"].to(dtype=weight_dtype)  # (B, V, 3, H, W); [0, 1]
        cam_poses = data["cam_poses_output"].to(dtype=weight_dtype)  # (B, V, 4)
        (B, V), device, dtype = images.shape[:2], images.device, images.dtype

        # SV3D
        cond_images = images[:, 0, ...]

        # Encode conditional input image (the first image) by CLIP image encoder
        with torch.no_grad():
            cond_image_embeds = self.sv3d.image_encoder(
                TF.normalize(
                    _resize_with_antialiasing(cond_images * 2.0 - 1.0, (224, 224)) * 0.5
                    + 0.5,
                    IMAGENET_DEFAULT_MEAN,
                    IMAGENET_DEFAULT_STD,
                ).to(self.sv3d.image_encoder.dtype)
            ).image_embeds.unsqueeze(
                1
            )  # (B, 1, D)

            # Encode conditional input image (the first image) by VAE encoder
            images = einops.rearrange(images, "b v c h w -> (b v) c h w")
            gt_latents = self.sv3d.vae.config.scaling_factor * (
                self.sv3d.vae.encode(images * 2.0 - 1.0).latent_dist.sample()
                if not self.opt.use_tiny_ae
                else self.sv3d.vae.encode(images * 2.0 - 1.0).latents
            )
            images = einops.rearrange(images, "(b v) c h w -> b v c h w", v=V)
            gt_latents = einops.rearrange(gt_latents, "(b v) c h w -> b v c h w", v=V)
            cond_latents = (
                gt_latents[:, 0, ...].unsqueeze(1).repeat(1, V, 1, 1, 1) / 0.18215
            )  # (B, V, 4, H', W')

        # Do classifier-free guidance
        if self.opt.do_classifier_free_guidance:
            negative_cond_image_embeds = torch.zeros_like(
                cond_image_embeds[0]
            ).unsqueeze(0)
            empty_prob = (
                torch.rand(B, device=device) < self.opt.conditioning_dropout_prob
            )
            cond_image_embeds[empty_prob, ...] = negative_cond_image_embeds

            negative_cond_latents = torch.zeros_like(cond_latents[0]).unsqueeze(0)
            empty_prob = (
                torch.rand(B, device=device) < self.opt.conditioning_dropout_prob
            )
            cond_latents[empty_prob, ...] = negative_cond_latents

        # Prepare camera conditions for SV3D
        if cam_poses is not None:
            polars_rad = cam_poses[:, :, 0]  # (B, V)
            azimuths_deg = torch.rad2deg(cam_poses[:, :, 1])  # (B, V)
            azimuths_deg = (
                azimuths_deg - azimuths_deg[:, 0:1]
            ) % 360.0  # [-180., +180] -> [0., 360]
            # azimuths_deg = torch.cat([azimuths_deg[:, 1:], azimuths_deg[:, 0:1]], dim=1)  # last frame is the conditional image
            azimuths_rad = torch.deg2rad(azimuths_deg)
        else:
            assert V == 24
            polars_rad = torch.deg2rad(
                90.0 - 10.0 * torch.ones((B, V), dtype=dtype, device=device)
            )
            azimuths_rad = torch.deg2rad(
                torch.linspace(0.0, 360.0, V + 1, dtype=dtype, device=device)[1:]
                % 360.0
            )
            azimuths_rad = einops.repeat(azimuths_rad, "v -> b v", b=B)
        added_time_ids = [
            1e-5 * torch.ones_like(polars_rad),
            polars_rad,
            azimuths_rad,
        ]

        # Last frame is the conditional image
        # gt_latents = torch.cat([gt_latents[:, 1:, ...], gt_latents[:, 0:1, ...]], dim=1)

        # Diffusion forward process
        noises = torch.randn_like(gt_latents)  # (B, V, 4, H', W')
        sigmas = rand_log_normal(shape=(B,), loc=0.7, scale=1.6).to(device)[
            :, None, None, None, None
        ]  # TODO: P_mean=0.7, P_std=1.6

        c_in = 1.0 / ((sigmas**2.0 + 1.0) ** 0.5)
        c_out = -sigmas / ((sigmas**2.0 + 1.0) ** 0.5)
        c_skip = 1.0 / (sigmas**2.0 + 1)
        c_noise = (sigmas.log() / 4.0).view(
            B,
        )
        loss_weights = (sigmas**2.0 + 1.0) / (sigmas**2.0)

        noisy_latents = gt_latents + sigmas * noises
        input_noisy_latents = torch.cat(
            [c_in * noisy_latents, cond_latents], dim=2
        )  # (B, V, 8, H', W')

        # Denoise
        model_output = self.sv3d.unet(
            input_noisy_latents,
            c_noise,
            encoder_hidden_states=cond_image_embeds,
            added_time_ids=added_time_ids,
        ).sample.detach()  # NOTE: detach here

        pred_original_latents = c_out * model_output + c_skip * noisy_latents

        # Last frame is the conditional image
        # pred_original_latents = torch.cat([pred_original_latents[:, -1:, ...], pred_original_latents[:, :-1, ...]], dim=1)
        # gt_latents = torch.cat([gt_latents[:, -1:, ...], gt_latents[:, :-1, ...]], dim=1)

        # LGM
        cam_poses_4x4 = data["cam_poses_4x4_output"].to(
            dtype=weight_dtype
        )  # (B, V, 4, 4)

        # Randomly choose `RANDOM_V`+1 views; one fixed view for the input conditional image
        RANDOM_V = 4 - 1  # TODO: make it configurable
        random_idxs = (torch.randperm(V - 1, device=device) + 1)[
            :RANDOM_V
        ].long()  # sample from [1, V-1]
        # Use gt conditional images for LGM
        latents = torch.cat(
            [
                gt_latents[:, 0:1, ...],
                pred_original_latents[:, random_idxs, ...],
            ],
            dim=1,
        )
        cam_poses_4x4 = torch.cat(
            [
                cam_poses_4x4[:, 0:1, ...],
                cam_poses_4x4[:, random_idxs, ...],
            ],
            dim=1,
        )

        render_outputs, _ = self.lgm.reconstruct(
            latents,
            cam_poses_4x4,
            data["cam_view"],
            data["cam_view_proj"],
            data["cam_pos"],
            input_latents=True,
            t=einops.repeat(c_noise, "b -> (b v)", v=4),  # `4` is hard-coded for LGM
        )

        pred_images = render_outputs["image"]  # (B, V', C, H, W)
        pred_alphas = render_outputs["alpha"]  # (B, V', 1, H, W)

        gt_images = data["images_output"]  # (B, V', 3, H, W)
        gt_masks = data["masks_output"]  # (B, V', 1, H, W)

        if vis:
            with torch.no_grad():
                # Denoised images by SV3D, for visualization
                pred_original_latents = einops.rearrange(
                    pred_original_latents, "b v c h w -> (b v) c h w"
                )
                denoised_images = self.sv3d.vae.decode(
                    1.0 / self.sv3d.vae.config.scaling_factor * pred_original_latents
                ).sample
                denoised_images = einops.rearrange(
                    denoised_images, "(b v) c h w -> b v c h w", v=V
                )
                denoised_images = (denoised_images * 0.5 + 0.5).clamp(0.0, 1.0)
        else:
            denoised_images = None

        # For visualization
        outputs["images_pred"] = pred_images
        outputs["images_denoised"] = denoised_images
        outputs["images_gt"] = gt_images
        # outputs["alphas_pred"] = pred_alphas
        # outputs["alphas_gt"] = gt_masks

        outputs["image_mse"] = image_mse = F.mse_loss(
            pred_images, gt_images, reduction="none"
        ).mean(
            (2, 3, 4)
        )  # (B, V)
        outputs["mask_mse"] = mask_mse = F.mse_loss(
            pred_alphas, gt_masks, reduction="none"
        ).mean(
            (2, 3, 4)
        )  # (B, V)
        recon_loss = image_mse + mask_mse  # (B, V)

        if self.opt.lambda_lpips > 0.0:
            lpips = self.lgm.lpips_loss(
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
            )
            lpips = einops.rearrange(lpips.squeeze(), "(b v) -> b v", v=V)  # (B, V)
            outputs["lpips"] = lpips.mean()
            recon_loss += self.opt.lambda_lpips * lpips
        else:
            with torch.no_grad():
                lpips = self.lgm.lpips_loss(
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
                )
                lpips = einops.rearrange(lpips.squeeze(), "(b v) -> b v", v=V)  # (B, V)
                outputs["lpips"] = lpips

        # Metric: PNSR
        with torch.no_grad():
            psnr = -10 * torch.log10(
                torch.mean((pred_images.detach() - gt_images) ** 2)
            )
            outputs["psnr"] = psnr

        recon_view_loss_weights = 1.0  # torch.tensor([10.,] + [1.] * (V-1), dtype=dtype, device=device).unsqueeze(0)  # (1, V); TODO: make it configurable
        recon_batch_loss_weights = torch.clamp(
            loss_weights.squeeze(), 1.0, 10.0
        ).unsqueeze(
            1
        )  # (B, 1); TODO: make it configurable
        outputs["recon_loss"] = recon_loss = (
            recon_view_loss_weights * recon_batch_loss_weights * recon_loss
        ).mean()

        # SV3D
        with torch.no_grad():
            pred_images = einops.rearrange(pred_images, "b v c h w -> (b v) c h w")
            pred_latents = self.sv3d.vae.config.scaling_factor * (
                self.sv3d.vae.encode(pred_images * 2.0 - 1.0).latent_dist.sample()
                if not self.opt.use_tiny_ae
                else self.sv3d.vae.encode(pred_images * 2.0 - 1.0).latents
            )
            pred_latents = einops.rearrange(
                pred_latents, "(b v) c h w -> b v c h w", v=V
            )

        # Last frame is the conditional image
        # gt_latents = torch.cat([gt_latents[:, 1:, ...], gt_latents[:, 0:1, ...]], dim=1)
        # pred_latents = torch.cat([pred_latents[:, 1:, ...], pred_latents[:, 0:1, ...]], dim=1)

        noisy_pred_latents = pred_latents + sigmas * noises
        input_noisy_pred_latents = torch.cat(
            [c_in * noisy_pred_latents, cond_latents], dim=2
        )  # (B, V, 8, H', W')

        # Denoise
        model_output = self.sv3d.unet(
            input_noisy_pred_latents,
            c_noise,
            encoder_hidden_states=cond_image_embeds,
            added_time_ids=added_time_ids,
        ).sample
        pred_original_latents = c_out * model_output + c_skip * noisy_pred_latents

        # Diffusion loss
        diffusion_batch_loss_weights = torch.clamp(
            loss_weights, 1.0, 10.0
        )  # (B, ...); TODO: make it configurable
        outputs["diffusion_loss"] = diffusion_loss = (
            diffusion_batch_loss_weights
            * F.mse_loss(pred_original_latents, gt_latents, reduction="none")
        ).mean()

        outputs["loss"] = diffusion_loss + recon_loss  # TODO: weighting

        return outputs

    @torch.no_grad()
    def evaluate(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):

        outputs = {}

        images = data["images_output"].to(dtype=weight_dtype)  # (B, V, 3, H, W); [0, 1]
        cam_poses = data["cam_poses_output"].to(dtype=weight_dtype)  # (B, V, 4)

        cond_images = images[:, 0, ...]
        cond_cam_poses = cam_poses
        verbose = False  # TODO: make it configurable

        assert torch.all(cond_images >= 0.0) and torch.all(cond_images <= 1.0)
        B, V = cond_images.shape[0], self.opt.num_views
        (H, W), S = cond_images.shape[-2:], self.opt.vae_scale_factor
        dtype, device = cond_images.dtype, cond_images.device

        # Encode conditional input image (the first image) by CLIP image encoder
        cond_image_embeds = self.sv3d.image_encoder(
            TF.normalize(
                _resize_with_antialiasing(cond_images * 2.0 - 1.0, (224, 224)) * 0.5
                + 0.5,
                IMAGENET_DEFAULT_MEAN,
                IMAGENET_DEFAULT_STD,
            )
        ).image_embeds.unsqueeze(
            1
        )  # (B, 1, D)

        # Encode conditional input image (the first image) by VAE encoder
        cond_latents = (
            self.sv3d.vae.encode(cond_images * 2.0 - 1.0).latent_dist.mode()
            if not self.opt.use_tiny_ae
            else (self.sv3d.vae.encode(cond_images * 2.0 - 1.0).latents / 0.18215)
        )
        cond_latents = cond_latents.unsqueeze(1).repeat(
            1, V, 1, 1, 1
        )  # (B, V, 4, H', W')

        # Do classifier-free guidance
        if self.opt.do_classifier_free_guidance:
            negative_cond_image_embeds = torch.zeros_like(cond_image_embeds)
            cond_image_embeds = torch.cat(
                [negative_cond_image_embeds, cond_image_embeds], dim=0
            )

            negative_cond_latents = torch.zeros_like(cond_latents)
            cond_latents = torch.cat([negative_cond_latents, cond_latents], dim=0)

        # Prepare camera conditions for SV3D
        if cond_cam_poses is not None:
            polars_rad = cond_cam_poses[:, :, 0]  # (B, V)
            azimuths_rad = cond_cam_poses[:, :, 1]  # (B, V)
            azimuths_rad = (azimuths_rad + np.deg2rad(360.0)) % np.deg2rad(
                360.0
            )  # [-180., +180] -> [0., 360]
            azimuths_rad = (azimuths_rad - azimuths_rad[:, 0:1]) % np.deg2rad(360.0)
            azimuths_rad = torch.cat(
                [azimuths_rad[:, 1:], azimuths_rad[:, 0:1]], dim=1
            )  # last frame is the conditional image
        else:
            assert V == 24
            polars_rad = torch.deg2rad(
                90.0 - torch.zeros((B, V), dtype=dtype, device=device)
            )
            azimuths_deg = np.linspace(0.0, 360.0, V + 1)[1:] % 360.0
            azimuths_rad = [
                np.deg2rad((a - azimuths_deg[-1]) % 360.0) for a in azimuths_deg
            ]  # last frame is the conditional image
            azimuths_rad[:-1].sort()
            azimuths_rad = torch.tensor(azimuths_rad, dtype=dtype, device=device)
            azimuths_rad = einops.repeat(azimuths_rad, "v -> b v", b=B)
        added_time_ids = [
            1e-5 * torch.ones_like(polars_rad),
            polars_rad,
            azimuths_rad,
        ]
        if self.opt.do_classifier_free_guidance:
            for i in range(3):
                added_time_ids[i] = torch.cat([added_time_ids[i]] * 2, dim=0)

        # Prepare timesteps
        self.sv3d.noise_scheduler.set_timesteps(
            self.opt.num_inference_timesteps, device=device
        )
        timesteps = self.sv3d.noise_scheduler.timesteps
        noise_scheduler_clone = deepcopy(
            self.sv3d.noise_scheduler
        )  # used for step from predicted original latents
        noise_scheduler_clone.config.prediction_type = "original_sample"
        noise_scheduler_clone.is_scale_input_called = True  # not really used

        # Prepare latent variables
        latents = torch.randn(B, V, 4, H // S, W // S, device=device, dtype=dtype)
        latents = latents * self.sv3d.noise_scheduler.init_noise_sigma

        # Prepare guidance scale (triangular scaling)
        guidance_scale = torch.cat(
            [
                torch.linspace(
                    self.opt.min_guidance_scale, self.opt.max_guidance_scale, V // 2
                ).unsqueeze(0),
                torch.linspace(
                    self.opt.max_guidance_scale, self.opt.min_guidance_scale, V - V // 2
                ).unsqueeze(0),
            ],
            dim=-1,
        )
        guidance_scale = guidance_scale.to(device, dtype=dtype)
        guidance_scale = guidance_scale.repeat(B, 1)
        while guidance_scale.ndim < latents.ndim:
            guidance_scale = guidance_scale.unsqueeze(-1)

        # Denoising loop
        for i, t in tqdm(
            enumerate(timesteps), total=len(timesteps), disable=(not verbose), ncols=125
        ):

            latent_model_input = (
                torch.cat([latents] * 2)
                if self.opt.do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.sv3d.noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = torch.cat([latent_model_input, cond_latents], dim=2)

            model_output = self.sv3d.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_image_embeds,
                added_time_ids=added_time_ids,
            ).sample

            # Perform guidance
            if self.opt.do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (
                    model_output_cond - model_output_uncond
                )

            pred_original_latents = self.sv3d.noise_scheduler.step(
                model_output, t, latents
            ).pred_original_sample

            # # Visualization intermedia generated images
            # pred_original_latents = einops.rearrange(pred_original_latents, "b v c h w -> (b v) c h w")
            # pred_original_images = self.sv3d.vae.decode(1. / self.sv3d.vae.config.scaling_factor * pred_original_latents).sample
            # pred_original_images = (pred_original_images * 0.5 + 0.5).clamp(0., 1.)  # (B, V, 3, H, W); [0, 1]
            # pred_original_images = einops.rearrange(pred_original_images, "(b v) c h w -> b v c h w", v=V)
            # pred_original_latents = einops.rearrange(pred_original_latents, "(b v) c h w -> b v c h w", v=V)
            # vis_batched_mvimages(pred_original_images, f"temp/sv3d_{i}.jpg")

            # Last frame is the conditional image
            # pred_original_latents = torch.cat([pred_original_latents[:, -1:, ...], pred_original_latents[:, :-1, ...]], dim=1)
            # latents = torch.cat([latents[:, -1:, ...], latents[:, :-1, ...]], dim=1)

            if True:  # TODO: think about it
                # LGM
                cam_poses_4x4 = data["cam_poses_4x4_output"].to(
                    dtype=weight_dtype
                )  # (B, V, 4, 4)
                # Use gt conditional images for LGM
                pred_original_latents_lgm = pred_original_latents.clone()
                pred_original_latents_lgm[:, 0, ...] = 0.18215 * (
                    cond_latents[B:, 0, ...]
                    if self.opt.do_classifier_free_guidance
                    else cond_latents[:, 0, ...]
                )

                # Choose 4 distinct views (last frame is the conditional image)
                random_idxs = torch.linspace(0, V, steps=4 + 1).long()[
                    :4
                ]  # `4` is hard-coded for LGM
                pred_original_latents_lgm = pred_original_latents_lgm[
                    :, random_idxs, ...
                ]
                cam_poses_4x4 = cam_poses_4x4[:, random_idxs, ...]

                render_outputs, _ = self.lgm.reconstruct(
                    pred_original_latents_lgm,
                    cam_poses_4x4,
                    data["cam_view"],
                    data["cam_view_proj"],
                    data["cam_pos"],
                    input_latents=True,
                    t=t,
                )

                pred_images = render_outputs["image"]  # (B, V, C, H, W)
                pred_images = einops.rearrange(pred_images, "b v c h w -> (b v) c h w")
                pred_latents = self.sv3d.vae.config.scaling_factor * (
                    self.sv3d.vae.encode(pred_images * 2.0 - 1.0).latent_dist.sample()
                    if not self.opt.use_tiny_ae
                    else self.sv3d.vae.encode(pred_images * 2.0 - 1.0).latents
                )
                pred_latents = einops.rearrange(
                    pred_latents, "(b v) c h w -> b v c h w", v=V
                )

                # # Visualization intermedia rendered images
                # pred_images_ = einops.rearrange(pred_images, "(b v) c h w -> b v c h w", v=V)
                # vis_batched_mvimages(pred_images_, f"temp/lgm_{i}.jpg")

                assert noise_scheduler_clone.config.prediction_type == "original_sample"
                latents = noise_scheduler_clone.step(
                    pred_latents, t, latents
                ).prev_sample

            else:
                # SV3D
                assert noise_scheduler_clone.config.prediction_type == "original_sample"
                latents = noise_scheduler_clone.step(
                    pred_original_latents, t, latents
                ).prev_sample

            # Last frame is the conditional image
            latents = torch.cat([latents[:, 1:, ...], latents[:, 0:1, ...]], dim=1)

        # # Last frame is the conditional image
        # latents = torch.cat([latents[:, -1:, ...], latents[:, :-1, ...]], dim=1)

        # latents = einops.rearrange(latents, "b v c h w -> (b v) c h w")
        # pred_images = self.sv3d.vae.decode(1. / self.sv3d.vae.config.scaling_factor * latents).sample
        pred_images = einops.rearrange(pred_images, "(b v) c h w -> b v c h w", v=V)
        # pred_images = (pred_images * 0.5 + 0.5).clamp(0., 1.)  # (B, V, 3, H, W); [0, 1]

        gt_images = data["images_output"]  # (B, V, 3, H, W)

        # For visualization; randomly choose min(V-1, 7) views; one fixed view for the input conditional image
        RANDOM_V = min(V - 1, 8 - 1)
        random_idxs = (torch.randperm(V - 1, device=device) + 1)[
            :RANDOM_V
        ].long()  # sample from [1, V-1]; TODO: make `7` configurable
        random_idxs = torch.concat(
            [torch.zeros_like(random_idxs)[0:1], random_idxs]
        ).long()
        outputs["images_pred"] = pred_images[:, random_idxs, ...]
        outputs["images_gt"] = gt_images[:, random_idxs, ...]

        # Evaluation metrics
        outputs["psnr"] = -10.0 * torch.log10(F.mse_loss(pred_images, gt_images))
        outputs["lpips"] = self.lgm.lpips_loss(
            # Downsampled to at most 256 to reduce memory cost
            F.interpolate(
                einops.rearrange(gt_images, "b v c h w -> (b v) c h w") * 2.0 - 1.0,
                (256, 256),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                einops.rearrange(pred_images, "b v c h w -> (b v) c h w") * 2.0 - 1.0,
                (256, 256),
                mode="bilinear",
                align_corners=False,
            ),
        ).mean()
        outputs["ssim"] = torch.tensor(
            calculate_ssim(
                (
                    einops.rearrange(gt_images, "b v c h w -> (b v c) h w")
                    .cpu()
                    .float()
                    .numpy()
                    * 255.0
                ).astype(np.uint8),
                (
                    einops.rearrange(pred_images, "b v c h w -> (b v c) h w")
                    .cpu()
                    .float()
                    .numpy()
                    * 255.0
                ).astype(np.uint8),
                channel_axis=0,
            ),
            device=gt_images.device,
        )

        return outputs
