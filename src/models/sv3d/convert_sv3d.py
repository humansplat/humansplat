import argparse

import numpy as np
import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.models.sv3d.pipeline_sv3d import SV3DPipeline
from src.models.sv3d.unet_spatio_temporal_condition import \
    UNetSpatioTemporalConditionModel

SVD_V1_CKPT = "stabilityai/stable-video-diffusion-img2vid-xt"
SD_V15_CKPT = "runwayml/stable-diffusion-v1-5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_ckpt_path",
        default="/tmp/lgm_setup/.cache/huggingface/hub/models--stabilityai--sv3d/snapshots/0941a09edb42171ac741d4760ecbe5a7bc2516a0/sv3d_p.safetensors",
        type=str,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--config_path",
        default="/opt/tiger/LGM/src/models/sv3d/sv3d_p.yaml",
        type=str,
        help="Config filepath.",
    )
    parser.add_argument(
        "--image_path",
        default="/opt/tiger/LGM/assets/openlrm/girl.png",
        type=str,
        help="Image filepath.",
    )
    parser.add_argument("--dump_path", default="/opt/tiger/LGM/out/sv3d", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    # original_ckpt = safetensors.torch.load_file(args.original_ckpt_path, device="cpu")

    # from omegaconf import OmegaConf
    # config = OmegaConf.load(args.config_path)

    # unet_config = create_unet_diffusers_config(config, image_size=768)

    # ori_config = unet_config.copy()
    # unet_config.pop("attention_head_dim")
    # unet_config.pop("use_linear_projection")
    # unet_config.pop("class_embed_type")
    # unet_config.pop("addition_embed_type")
    # unet = UNetSpatioTemporalConditionModel(**unet_config)
    # unet_state_dict = convert_ldm_unet_checkpoint(original_ckpt, ori_config)
    # unet.load_state_dict(unet_state_dict, strict=True)
    # unet.save_pretrained(args.dump_path, push_to_hub=args.push_to_hub)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "chenguolin/sv3d", low_cpu_mem_usage=False
    )

    vae = AutoencoderKL.from_pretrained(SD_V15_CKPT, subfolder="vae", variant="fp16")
    scheduler = EulerDiscreteScheduler.from_pretrained(
        SVD_V1_CKPT, subfolder="scheduler", variant="fp16"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        SVD_V1_CKPT, subfolder="image_encoder", variant="fp16"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        SVD_V1_CKPT, subfolder="feature_extractor", variant="fp16"
    )

    pipeline = SV3DPipeline(
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        scheduler=scheduler,
    )
    # pipeline.save_pretrained(args.dump_path, push_to_hub=args.push_to_hub)

    num_frames = 21
    sv3d_res = 576
    elevations_deg = [10] * num_frames
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    pipeline = pipeline.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", enabled=True):
            image = Image.open(args.image_path)
            image.load()  # required for `.split()`
            if len(image.split()) == 4:  # RGBA
                input_image = Image.new(
                    "RGB", image.size, (255, 255, 255)
                )  # pure white
                input_image.paste(
                    image, mask=image.split()[3]
                )  # `3` is the alpha channel
            else:
                input_image = image
            video_frames = pipeline(
                input_image.resize((sv3d_res, sv3d_res)),
                height=sv3d_res,
                width=sv3d_res,
                num_frames=num_frames,
                decode_chunk_size=8,
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
            ).frames[0]

    export_to_gif(video_frames, "temp.gif")
