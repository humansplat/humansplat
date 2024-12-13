from typing import *

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import (AutoencoderKL, AutoencoderKLTemporalDecoder,
                       AutoencoderTiny, EulerDiscreteScheduler)
from diffusers.utils.import_utils import is_xformers_available
from kiui.lpips import LPIPS
from skimage.metrics import structural_similarity as calculate_ssim
from torch import Tensor, nn
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from src.models.sv3d.pipeline_sv3d import _resize_with_antialiasing
from src.models.sv3d.unet_spatio_temporal_condition import \
    UNetSpatioTemporalConditionModel
from src.options import Options

SVD_V1_CKPT = "stabilityai/stable-video-diffusion-img2vid-xt"
SD_V15_CKPT = "runwayml/stable-diffusion-v1-5"

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


class SV3D(nn.Module):
    def __init__(self, opt: Options, weight_dtype=torch.bfloat16):
        super().__init__()
        self.opt = opt
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            SVD_V1_CKPT, subfolder="scheduler"
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            SVD_V1_CKPT, subfolder="image_encoder", variant="fp16"
        )
        if opt.use_tiny_ae:
            self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
        else:
            # self.vae = AutoencoderKLTemporalDecoder.from_pretrained(SVD_V1_CKPT, subfolder="vae", variant="fp16")
            self.vae = AutoencoderKL.from_pretrained(
                SD_V15_CKPT, subfolder="vae", variant="fp16"
            )
        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(
            "paulpanwang/GenHuman3D", subfolder="unet", variant="fp16"
        )
        # self.feature_extractor = CLIPImageProcessor.from_pretrained(
        #     SVD_V1_CKPT, subfolder="feature_extractor",
        # )

        # For mixed precision training we cast the weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required
        self.vae.to(dtype=weight_dtype)
        self.image_encoder.to(dtype=weight_dtype)

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # Evaluation metrics
        self.enable_lpips = opt.sv3d_enable_lpips
        if self.enable_lpips:
            self.lpips_loss = LPIPS(net="vgg").to(self.image_encoder.device)
            self.lpips_loss.requires_grad_(False)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()
            if isinstance(self.unet, AutoencoderKL):
                self.vae.enable_slicing()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if "
            "doing mixed precision training, copy of the weights should still be float32."
        )
        if self.unet.dtype != torch.float32:
            raise ValueError(
                f"UNet loaded as datatype {self.unet.dtype}. {low_precision_error_string}"
            )

        self.unet.enable_gradient_checkpointing()  # to save memory
        torch.backends.cuda.matmul.allow_tf32 = True

        num_transformer_blocks = 0
        self.unet.requires_grad_(False)

        # transformer_blocks
        for name, param in self.unet.named_parameters():
            if "temporal_transformer_blocks" in name:
                num_transformer_blocks += 1

        num_finetune_block = 0
        for name, param in self.unet.named_parameters():
            if "temporal_transformer_blocks" in name:
                if num_finetune_block >= num_transformer_blocks - 100:
                    param.requires_grad_(True)
                num_finetune_block += 1

        for name, param in self.unet.named_parameters():
            if "add_embedding" in name:  #  finetune the camera embbeding part
                param.requires_grad_(True)

    def state_dict(self, **kwargs):
        # Remove frozen pretrained parameters
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k or "vae" in k or "image_encoder" in k:
                del state_dict[k]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def compute_loss(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):
        self.vae.eval()
        self.image_encoder.eval()
        outputs = {}
        images = data["images_output"].to(dtype=weight_dtype)  # (B, V, 3, H, W); [0, 1]
        cam_poses = data["cam_poses_output"].to(dtype=weight_dtype)  # (B, V, 4)
        _, return_dict = self.diffuse_and_denoise(images, cam_poses)
        outputs["loss"] = return_dict["loss"]
        return outputs

    @torch.no_grad()
    def evaluate(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):
        outputs = {}
        images = data["images_output"].to(dtype=weight_dtype)  # (B, V, 3, H, W); [0, 1]
        cam_poses = data["cam_poses_output"].to(dtype=weight_dtype)  # (B, V, 4)
        V = self.opt.num_views
        pred_images, _ = self.generate(images[:, 0, ...], cam_poses)
        gt_images = data["images_output"]  # (B, V, 3, H, W)

        outputs["images_pred"] = pred_images[:, ::2, ...]
        outputs["images_gt"] = gt_images[:, ::2, ...]
        outputs["psnr"] = -10.0 * torch.log10(F.mse_loss(pred_images, gt_images))

        outputs["ssim"] = torch.tensor(
            calculate_ssim(
                (
                    einops.rearrange(gt_images, "b v c h w -> (b v) h w c")
                    .cpu()
                    .float()
                    .numpy()
                    * 255.0
                ).astype(np.uint8),
                (
                    einops.rearrange(pred_images, "b v c h w -> (b v) h w c")
                    .cpu()
                    .float()
                    .numpy()
                    * 255.0
                ).astype(np.uint8),
                channel_axis=3,
            ),
            device=gt_images.device,
        )

        # disable lpips in sv3d_p
        if self.enable_lpips:
            outputs["lpips"] = self.lpips_loss(
                # Downsampled to at most 224 to reduce memory cost
                F.interpolate(
                    einops.rearrange(
                        gt_images.to(torch.float32), "b v c h w -> (b v) c h w"
                    )
                    * 2.0
                    - 1.0,
                    (112, 112),
                    mode="bilinear",
                    align_corners=False,
                ),
                F.interpolate(
                    einops.rearrange(
                        pred_images.to(torch.float32), "b v c h w -> (b v) c h w"
                    )
                    * 2.0
                    - 1.0,
                    (112, 112),
                    mode="bilinear",
                    align_corners=False,
                ),
            ).mean()
        else:
            outputs["lpips"] = 1 - outputs["ssim"]

        return outputs

    def diffuse_and_denoise_bak(
        self,
        images: Tensor,  # in [0, 1]
        cam_poses: Optional[Tensor] = None,  # (polar, azimuth, ...)
        noises: Optional[Tensor] = None,
        gt_latents: Optional[Tensor] = None,
        sigmas: Optional[Tensor] = None,
        do_classifier_free_guidance: Optional[bool] = None,
        cond_images: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        assert torch.all(images >= 0.0) and torch.all(images <= 1.0)
        (B, V), device, dtype = images.shape[:2], images.device, images.dtype

        do_classifier_free_guidance = (
            do_classifier_free_guidance or self.opt.do_classifier_free_guidance
        )
        # Encode conditional input image (the first image) by CLIP image encoder
        with torch.no_grad():
            if cond_images is None:
                cond_images = images[:, 0, ...]
            else:
                assert torch.all(cond_images >= 0.0) and torch.all(cond_images <= 1.0)

            cond_image_embeds = self.image_encoder(
                TF.normalize(
                    _resize_with_antialiasing(cond_images * 2.0 - 1.0, (224, 224)) * 0.5
                    + 0.5,
                    IMAGENET_DEFAULT_MEAN,
                    IMAGENET_DEFAULT_STD,
                ).to(self.image_encoder.dtype)
            ).image_embeds.unsqueeze(
                1
            )  # (B, 1, D)

            # Encode conditional input image (the first image) by VAE encoder
            images = einops.rearrange(images, "b v c h w -> (b v) c h w")

            latents = self.vae.config.scaling_factor * (
                self.vae.encode(images * 2.0 - 1.0).latent_dist.sample()
                if not self.opt.use_tiny_ae
                else self.vae.encode(images * 2.0 - 1.0).latents
            )

            images = einops.rearrange(images, "(b v) c h w -> b v c h w", v=V)
            latents = einops.rearrange(latents, "(b v) c h w -> b v c h w", v=V)
            if gt_latents is None:  # use latents of input images as ground-truth
                gt_latents = latents
            cond_latents = (
                gt_latents[:, 0, ...].unsqueeze(1).repeat(1, V, 1, 1, 1)
                / self.vae.config.scaling_factor
            )  # (B, V, 4, H', W')

            # Do classifier-free guidance
            if do_classifier_free_guidance:
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

            # Diffusion forward process
            if noises is None:
                noises = torch.randn_like(latents)  # (B, V, 4, H', W')

            if sigmas is None:
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
            noisy_latents = latents + sigmas * noises
            input_noisy_latents = torch.cat(
                [c_in * noisy_latents, cond_latents], dim=2
            )  # (B, V, 8, H', W')

        # Denoise
        if noises is not None:
            model_output = self.unet(
                input_noisy_latents,
                c_noise,
                encoder_hidden_states=cond_image_embeds,
                added_time_ids=added_time_ids,
            ).sample
            pred_original_latents = c_out * model_output + c_skip * noisy_latents

        # Diffusion loss
        loss = (
            loss_weights
            * F.mse_loss(pred_original_latents, gt_latents, reduction="none")
        ).mean()

        return pred_original_latents, {
            "loss": loss,
            "latents": latents,
            "noises": noises,
            "sigmas": sigmas,
            "loss_weights": loss_weights,
        }

    def diffuse_and_denoise(
        self,
        images: Tensor,  # in [0, 1]
        cam_poses: Optional[Tensor] = None,  # (polar, azimuth, ...)
        noises: Optional[Tensor] = None,
        gt_latents: Optional[Tensor] = None,
        sigmas: Optional[Tensor] = None,
        do_classifier_free_guidance: Optional[bool] = None,
        cond_images: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        assert torch.all(images >= 0.0) and torch.all(images <= 1.0)
        (B, V), device, dtype = images.shape[:2], images.device, images.dtype

        do_classifier_free_guidance = (
            do_classifier_free_guidance or self.opt.do_classifier_free_guidance
        )
        # Encode conditional input image (the first image) by CLIP image encoder
        with torch.no_grad():
            if cond_images is None:
                cond_images = images[:, 0, ...]
            else:
                assert torch.all(cond_images >= 0.0) and torch.all(cond_images <= 1.0)

            cond_image_embeds = self.image_encoder(
                TF.normalize(
                    _resize_with_antialiasing(cond_images * 2.0 - 1.0, (224, 224)) * 0.5
                    + 0.5,
                    IMAGENET_DEFAULT_MEAN,
                    IMAGENET_DEFAULT_STD,
                ).to(self.image_encoder.dtype)
            ).image_embeds.unsqueeze(
                1
            )  # (B, 1, D)

            # Encode conditional input image (the first image) by VAE encoder
            images = einops.rearrange(images, "b v c h w -> (b v) c h w")

            latents = self.vae.config.scaling_factor * (
                self.vae.encode(images * 2.0 - 1.0).latent_dist.sample()
                if not self.opt.use_tiny_ae
                else self.vae.encode(images * 2.0 - 1.0).latents
            )

            images = einops.rearrange(images, "(b v) c h w -> b v c h w", v=V)
            latents = einops.rearrange(latents, "(b v) c h w -> b v c h w", v=V)
            if gt_latents is None:  # use latents of input images as ground-truth
                gt_latents = latents
            cond_latents = (
                gt_latents[:, 0, ...].unsqueeze(1).repeat(1, V, 1, 1, 1)
                / self.vae.config.scaling_factor
            )  # (B, V, 4, H', W')
        return latents, {}

    @torch.no_grad()
    def _decode_latents(
        self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14
    ):
        latents = 1 / self.vae.config.scaling_factor * latents
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if isinstance(self.vae, AutoencoderKLTemporalDecoder):
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(
                latents[i : i + decode_chunk_size], **decode_kwargs
            ).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        return frames

    @torch.no_grad()
    def generate(
        self,
        cond_images: Tensor,  # (B, 3, H, W) in [0, 1]
        cond_cam_poses: Optional[Tensor] = None,  # (B, V, 4); (polar, azimuth, ...)
        do_classifier_free_guidance: Optional[bool] = None,
        verbose: bool = False,
    ):
        assert torch.all(cond_images >= 0.0) and torch.all(cond_images <= 1.0)
        B, V = cond_images.shape[0], self.opt.num_views
        (H, W), S = cond_images.shape[-2:], self.opt.vae_scale_factor
        dtype, device = cond_images.dtype, cond_images.device

        do_classifier_free_guidance = (
            do_classifier_free_guidance or self.opt.do_classifier_free_guidance
        )

        # Encode conditional input image (the first image) by CLIP image encoder
        cond_image_embeds = self.image_encoder(
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
            self.vae.encode(cond_images * 2.0 - 1.0).latent_dist.mode()
            if not self.opt.use_tiny_ae
            else (self.vae.encode(cond_images * 2.0 - 1.0).latents / 0.18215)
        )
        cond_latents = cond_latents.unsqueeze(1).repeat(
            1, V, 1, 1, 1
        )  # (B, V, 4, H', W')

        # Do classifier-free guidance
        if do_classifier_free_guidance:
            negative_cond_image_embeds = torch.zeros_like(cond_image_embeds)
            cond_image_embeds = torch.cat(
                [negative_cond_image_embeds, cond_image_embeds], dim=0
            )
            negative_cond_latents = torch.zeros_like(cond_latents)
            cond_latents = torch.cat([negative_cond_latents, cond_latents], dim=0)

        # Prepare camera conditions for SV3D
        if cond_cam_poses is not None:
            polars_rad = cond_cam_poses[:, :, 0]  # (B, V)
            azimuths_deg = torch.rad2deg(cond_cam_poses[:, :, 1])  # (B, V)
            azimuths_deg = (
                azimuths_deg - azimuths_deg[:, 0:1]
            ) % 360.0  # [-180., +180] -> [0., 360]
            # azimuths_deg = torch.cat([azimuths_deg[:, 1:], azimuths_deg[:, 0:1]], dim=1)  # last frame is the conditional image
            azimuths_rad = torch.deg2rad(azimuths_deg)
        else:
            # assert V == 24
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
        if do_classifier_free_guidance:
            for i in range(3):
                added_time_ids[i] = torch.cat([added_time_ids[i]] * 2, dim=0)

        # Prepare timesteps
        self.noise_scheduler.set_timesteps(
            self.opt.num_inference_timesteps, device=device
        )
        timesteps = self.noise_scheduler.timesteps

        # Prepare latent variables
        latents = torch.randn(B, V, 4, H // S, W // S, device=device, dtype=dtype)
        latents = latents * self.noise_scheduler.init_noise_sigma

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
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = torch.cat([latent_model_input, cond_latents], dim=2)

            model_output = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_image_embeds,
                added_time_ids=added_time_ids,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * (
                    model_output_cond - model_output_uncond
                )
            latents = self.noise_scheduler.step(model_output, t, latents).prev_sample

        # Last frame is the conditional image
        # latents = torch.cat([latents[:, -1:, ...], latents[:, :-1, ...]], dim=1)
        latents = einops.rearrange(latents, "b v c h w -> (b v) c h w")
        pred_images = self._decode_latents(
            latents=latents, num_frames=V, decode_chunk_size=1
        )
        latents = einops.rearrange(latents, "(b v) c h w -> b v c h w", v=V)
        pred_images = einops.rearrange(pred_images, "(b v) c h w -> b v c h w", v=V)
        pred_images = (pred_images * 0.5 + 0.5).clamp(
            0.0, 1.0
        )  # (B, V, 3, H, W); [0, 1]

        torch.cuda.empty_cache()

        return pred_images, {
            "latents": latents,
        }
