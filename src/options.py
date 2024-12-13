from dataclasses import dataclass
from typing import *


@dataclass
class Options:
    # Diffusion
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 25
    do_classifier_free_guidance: bool = True
    conditioning_dropout_prob: float = 0.05
    guidance_scale: float = 5.0
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 2.5

    # Latent-to-pixel LGM
    latent_to_pixel: bool = True
    use_tiny_ae: bool = False
    vae_scale_factor: int = 8
    load_smpl: bool = True
    # LGM parameters
    backbone_type: str = "unet"
    # Input image size for encoder or GS UNet
    input_size: int = 256
    # UNet parameters
    down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024, 1024)
    down_attention: Tuple[bool, ...] = (False, False, False, True, True, True)
    mid_attention: bool = True
    up_channels: Tuple[int, ...] = (1024, 1024, 512, 256)
    up_attention: Tuple[bool, ...] = (True, True, True, False)
    # UViT parameters
    patch_size: int = 16
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    last_conv: bool = True
    skip: bool = True  # otherwise, it's a vanilla ViT
    # UNet output size, dependent on `input_size` and the U-Net structure
    splat_size: int = 64
    # Gaussian rendering size and ground-truth image size
    output_size: int = 256
    render_size: int = 256
    render_type: str = "default"
    gs_patch_size: int = 192
    # LPIPS loss weight
    lambda_lpips: float = 0.1

    ## Dataset
    # Load sorted multi-view images as a video
    load_sorted_images: bool = False
    # ImageNet statistics normalization
    imagenet_stats_norm: bool = False
    # Not include input views for supervision
    exclude_input_views: bool = False
    random_load = True


    # Human dataset
    thuman2_dir: str = "data/Thuman2/"
    render_2k2k_dir: str = "data/2K2K/"
    render_twindom_dir: str = "data/Twindom"

    shuffle_buffer_size: int = 1000
    # FoVy of the dataset
    fovy: float = 39.6
    # Camera near plane (not important)
    znear: float = 0.01
    # Camera far plane (not important)
    zfar: float = 1000
    # Number of all views (input + output)
    num_views: int = 8
    # Number of input views
    num_input_views: int = 4
    # Camera radius for camera normalization
    cam_radius: float = 1.5
    # Augmentation probability for grid distortion
    prob_grid_distortion: float = 0.5
    # Augmentation probability for camera jitter
    prob_cam_jitter: float = 0.5

    # TODO: ablation on  window_size
    k_window_size = 2
    smpl_emb_dim = 512

    norm_camera: bool = True
    norm_radius: float = 1.4  # the min distance in GObjaverse (cf. `RichDreamer` Sec. 3.1); only used when `norm_camera` is True

    # hyper
    load_normal = False
    load_caption = False
    use_checkpoint = False
    sv3d_enable_lpips = False

# Set all settings for different tasks and model size
opt_dict: Dict[str, Options] = {}
opt_docs: Dict[str, str] = {}

opt_docs["small"] = "the default settings for LGM, i.e., small"
opt_dict["small"] = Options()

opt_docs["big"] = "big model with higher resolution Gaussians"
opt_dict["big"] = Options(
    input_size=256,
    up_channels=(1024, 1024, 512, 256, 128),  # one more layer
    up_attention=(True, True, True, False, False),
    splat_size=128,
    output_size=512,  # render & supervise Gaussians at a higher resolution
)

opt_docs["tiny"] = "tiny model for ablation"
opt_dict["tiny"] = Options(
    input_size=256,
    down_channels=(32, 64, 128, 256, 512),
    down_attention=(False, False, False, False, True),
    up_channels=(512, 256, 128),
    up_attention=(True, False, False),
    splat_size=64,
    output_size=256,
)

opt_docs["tiny_uvit"] = "default UViT model for LGM"
opt_dict["tiny_uvit"] = Options(
    backbone_type="uvit",
    input_size=256,
    embed_dim=512,
    depth=12,
    num_heads=8,
    skip=True,
    splat_size=64,
    output_size=256,
)

opt_docs[
    "tiny_uvit_l2p_576"
] = "default UViT model for latent-to-pixel LGM on 576x576 resolution"

opt_dict["tiny_uvit_l2p_576"] = Options(
    latent_to_pixel=True,
    use_tiny_ae=False,
    backbone_type="uvit",
    input_size=576,
    patch_size=32,
    embed_dim=512,
    depth=12,
    num_heads=8,
    skip=True,
    splat_size=72,
    output_size=576,
    render_size=576,
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
)

opt_docs[
    "tiny_uvit_l2p_256"
] = "default UViT model for latent-to-pixel LGM on 256x256 resolution"
opt_dict["tiny_uvit_l2p_256"] = Options(
    latent_to_pixel=True,
    use_tiny_ae=False,
    backbone_type="uvit",
    input_size=256,
    patch_size=8,
    embed_dim=512,
    depth=12,
    num_heads=8,
    skip=True,
    splat_size=64,
    output_size=256,
    render_size=256,
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
)

opt_docs["sv3d_24"] = "finetune SV3D_p on 24 frames"
opt_dict["sv3d_24"] = Options(
    use_tiny_ae=False,
    input_size=512,
    output_size=512,
    load_sorted_images=True,
    num_views=18,
    num_input_views=1,
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
)


opt_docs[
    "noise_lgm"
] = "default model for noise LGM that reconstructs from SV3D denoised latents"

opt_dict["noise_lgm"] = Options(
    latent_to_pixel=True,
    use_tiny_ae=True,  # TODO: use standard stable diffusion VAE
    backbone_type="uvit",
    input_size=576,
    patch_size=32,
    embed_dim=512,
    depth=12,
    num_heads=8,
    skip=True,
    splat_size=72,
    output_size=576,
    render_size=576,
    render_type="default",
    gs_patch_size=72,
    load_sorted_images=True,
    num_views=21, # 21 for 24 frames
    num_input_views=1,  # not really used
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
)

opt_dict["sv3d_24"] = Options(
    use_tiny_ae=False,
    input_size=512,
    output_size=512,
    load_sorted_images=True,
    num_views=18,
    num_input_views=1,
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
)


opt_dict["humansplat"] = Options(
    latent_to_pixel=True,
    use_tiny_ae=False,
    backbone_type="uvit",
    input_size=512,
    patch_size=8,
    embed_dim=64,
    depth=6,
    num_heads=8,
    skip=True,
    splat_size=128,
    output_size=512,
    render_size=512,
    render_type="default",
    gs_patch_size=64,
    load_sorted_images=False,
    num_views=12,
    num_input_views=4,
    prob_grid_distortion=0.0,
    prob_cam_jitter=0.0,
    load_smpl=False,
)
