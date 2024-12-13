from typing import *

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderTiny
from kiui.lpips import LPIPS
from torch import Tensor

from src.options import Options
from src.utils.op_util import get_rays_batch

from .networks.gs import GaussianRenderer
from .networks.unet import UNet
from .networks.uvit import UViT


class LGM(nn.Module):
    def __init__(self, opt: Options, weight_dtype=torch.bfloat16, use_t_cond=False):
        super().__init__()

        self.opt = opt

        if opt.latent_to_pixel:
            if opt.use_tiny_ae:
                self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", subfolder="vae", variant="fp16"
                )
            self.vae.decoder = None  # not used; to save memory

            # For mixed precision training we cast the vae weights to half-precision
            # as these models are only used for inference, keeping weights in full precision is not required
            self.vae.to(dtype=weight_dtype)

            self.vae.requires_grad_(False)

        if opt.load_caption:
            self.text_encoder = CLIPTextEncoder("openai/clip-vit-large-patch14")
            self.text_encoder.to(dtype=weight_dtype)
            self.text_encoder.requires_grad_(False)

        in_chanes = 9 if not self.opt.latent_to_pixel else 10  # input pixel or latent
        if self.opt.load_normal:
            in_chanes += in_chanes - 6  # `6` for plucker embeddings

        if opt.backbone_type == "unet":
            # UNet
            self.model = UNet(
                in_chanes,
                14,
                down_channels=opt.down_channels,
                down_attention=opt.down_attention,
                mid_attention=opt.mid_attention,
                up_channels=opt.up_channels,
                up_attention=opt.up_attention,
                num_frames=4,  # TODO: hard-coded for LGM
            )
        elif opt.backbone_type == "uvit":
            self.model = UViT(
                in_chanes,
                14,
                img_size=opt.input_size
                // (1 if not self.opt.latent_to_pixel else self.opt.vae_scale_factor),
                patch_size=opt.patch_size
                // (1 if not self.opt.latent_to_pixel else self.opt.vae_scale_factor),
                splat_size=opt.splat_size,
                embed_dim=opt.embed_dim,
                depth=opt.depth,
                num_heads=opt.num_heads,
                conv=opt.last_conv,
                skip=opt.skip,
                use_mvattn=True,
                num_frames=4,  # TODO: hard-coded for LGM
                use_t_cond=use_t_cond,
                use_checkpoint=opt.use_checkpoint,
                use_cross_attn=opt.load_caption,
                dim_kv=self.text_encoder.text_emb_dim
                if self.opt.load_caption
                else None,
            )
        else:
            raise NotImplementedError(
                f"Invalid `opt.backbone_type`: [{self.opt.backbone_type}]"
            )

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activations
        self.pos_act = lambda x: x.clamp(
            -0.5, 0.5
        )  # GObjaverse normalizes objects in [-0.5, 0.5]^3
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: torch.sigmoid(x)

        # LPIPS loss
        if opt.lambda_lpips > 0.0:
            self.lpips_loss = LPIPS(net="vgg")
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # Remove frozen pretrained parameters
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k or "vae" in k or "text_encoder" in k:
                del state_dict[k]
        return state_dict

    def forward(self, *args, func_name="compute_loss", **kwargs):
        # To support different forward functions for models wrapped by `accelerate`
        return getattr(self, func_name)(*args, **kwargs)

    def forward_gaussians(
        self,
        images: Tensor,
        t: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ):
        B, V, C, H, W = images.shape
        images = images.view(B * V, C, H, W)

        x = self.model(images, t, context)  # (B*V, 14, H, W)

        x = x.reshape(B, V, -1, self.opt.splat_size, self.opt.splat_size)
        assert x.shape[2] == 14
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, x.shape[2])

        pos = self.pos_act(x[..., 0:3])  # (B, N, 3)
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat(
            [pos, opacity, scale, rotation, rgbs], dim=-1
        )  # (B, N, 14)
        return gaussians

    def compute_loss(self, data: Dict[str, Tensor], weight_dtype=torch.bfloat16):
        outputs = {}

        images = data["images_input"].to(dtype=weight_dtype)  # (B, V_in, 3, H, W)
        cam_poses_4x4_input = data["cam_poses_4x4_input"].to(
            dtype=weight_dtype
        )  # (B, V_in, 4, 4)

        if self.opt.load_caption:
            captions = data["caption"]  # (B,); list of stirngs
        else:
            captions = None

        render_outputs, _ = self.reconstruct(
            images,
            cam_poses_4x4_input,
            data["cam_view"],
            data["cam_view_proj"],
            data["cam_pos"],
            captions=captions,
        )

        pred_images = render_outputs["image"]  # (B, V, C, H, W)
        pred_alphas = render_outputs["alpha"]  # (B, V, 1, H, W)

        gt_images = data["images_output"]  # (B, V, 3, H, W)
        if self.opt.load_normal:
            gt_images = gt_images[:, :, :3, ...]  # only use rgb for supervision
        gt_masks = data["masks_output"]  # (B, V, 1, H, W)

        # For visualization
        outputs["images_pred"] = pred_images
        outputs["images_gt"] = gt_images
        # outputs["alphas_pred"] = pred_alphas
        # outputs["alphas_gt"] = gt_masks

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

        # Metric: PNSR
        with torch.no_grad():
            psnr = -10 * torch.log10(
                torch.mean((pred_images.detach() - gt_images) ** 2)
            )
            outputs["psnr"] = psnr

        return outputs

    def reconstruct(
        self,
        images: Tensor,  # normalized
        cam_poses_4x4_input: Tensor,
        # For GaussianRenderer
        cam_view: Tensor,  # (B, V, 4, 4); w2c in COLMAP camera convention
        cam_view_proj: Tensor,  # (B, V, 4, 4); w2c @ c2i -> w2i
        cam_pos: Tensor,  # (B, V, 3)
        input_latents: bool = False,
        t: Optional[Tensor] = None,
        captions: Optional[Tensor] = None,
    ):
        V_in = images.shape[1]

        if self.opt.load_caption:
            with torch.no_grad():
                context, _ = self.text_encoder(captions)  # (B, N, D)
        else:
            context = None

        if self.opt.latent_to_pixel and (not input_latents):
            # Encode images to latents
            images = einops.rearrange(images, "b v c h w -> (b v) c h w")

            if self.opt.load_normal:
                images, normals = images.chunk(
                    2, dim=1
                )  # (B*V, 3, H, W), (B*V, 3, H, W)
                images = torch.cat([images, normals], dim=0)  # (2*B*V, 3, H, W)

            with torch.no_grad():
                images = self.vae.config.scaling_factor * (
                    self.vae.encode(images).latent_dist.sample()
                    if not self.opt.use_tiny_ae
                    else self.vae.encode(images).latents
                )

            if self.opt.load_normal:
                images, normals = images.chunk(
                    2, dim=0
                )  # (B*V, 4, H, W), (B*V, 4, H, W)
                images = torch.cat([images, normals], dim=1)  # (B*V, 8, H, W)

            images = einops.rearrange(images, "(b v) c h w -> b v c h w", v=V_in)

        # Ray embeddings
        ray_embeddings = []
        for i in range(V_in):
            rays_o, rays_d = get_rays_batch(
                cam_poses_4x4_input[:, i, ...],
                self.opt.input_size,
                self.opt.input_size,
                self.opt.fovy,
            )
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
            )  # (B, H, W, 6)
            ray_embeddings.append(rays_plucker)
        ray_embeddings = (
            torch.stack(ray_embeddings, dim=1).permute(0, 1, 4, 2, 3).contiguous()
        )  # (B, V_in, 6, H, W)
        # Resize embedding size to optional different image/latent size
        ray_embeddings = einops.rearrange(ray_embeddings, "b v c h w -> (b v) c h w")
        ray_embeddings = F.interpolate(
            ray_embeddings, images.shape[-2:], mode="bilinear", align_corners=False
        )
        ray_embeddings = einops.rearrange(
            ray_embeddings, "(b v) c h w -> b v c h w", v=V_in
        )

        images = torch.cat([images, ray_embeddings], dim=2)  # (B, V_in, 9 or 10, H, W)

        # Use the input views to predict Gaussians
        gaussians = self.forward_gaussians(images, t, context)  # (B, N, 14)

        # Always use white bg
        bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # Use all views for rendering and supervision
        render_outputs = self.gs.render(
            gaussians, cam_view, cam_view_proj, cam_pos, bg_color=bg_color
        )

        return render_outputs, {"gaussians": gaussians}

    def prepare_default_rays(self, elevation=0.0):
        from kiui.cam import orbit_camera

        from src.utils.op_util import get_rays

        cam_poses = []
        for i in range(4):  # TODO: hard-coded for LGM
            cam_poses.append(
                orbit_camera(elevation, 360 * i / 4, radius=self.opt.cam_radius)
            )
        cam_poses = torch.from_numpy(np.stack(cam_poses, axis=0))  # (V, 4, 4)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(
                cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy
            )
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1
            )  # (H, W, 6)
            rays_embeddings.append(rays_plucker)
        rays_embeddings = (
            torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()
        )  # (V, 6, H, W)
        return rays_embeddings
