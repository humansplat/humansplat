import os

import numpy as np
import torch
from PIL import Image

from src.models import SV3D
from src.options import opt_dict
from src.utils.vis_util import tensor_to_gif


def rembg_and_center_wrapper(image_path: str, image_size: int) -> str:
    """Run `rembg_and_center.py` to remove background and center the image, and return
    the path to the new image.
    """
    os.system(
        f"python3 extensions/rembg_and_center.py {image_path} --size {image_size}"
    )
    directory, _ = os.path.split(image_path)
    file_base = os.path.basename(image_path).split(".")[0]
    new_filename = f"{file_base}_rgba.png"
    new_image_path = os.path.join(directory, new_filename)
    return new_image_path


from loguru import logger

opt = opt_dict["sv3d_24"]
opt.num_views = 18
opt.num_inference_timesteps = 20

sv3d = SV3D(opt).to(dtype=torch.float32, device="cuda")
from loguru import logger

hdfs_dir = "data/aigc/ckpt/humanLGM/sv3d_512_finetune"

infer_from_iter = 140000

ckpt_dir = "out"
from safetensors.torch import load_model as safetensors_load_model

if not os.path.exists(f"{ckpt_dir}/{infer_from_iter}"):
    os.system(
        f"hdfs dfs -get {os.path.join(hdfs_dir, f'{infer_from_iter:06d}.tar')} {ckpt_dir} && "
        + f"tar -xf {os.path.join(ckpt_dir, f'{infer_from_iter:06d}.tar')} -C {ckpt_dir} && "
        + f"rm {os.path.join(ckpt_dir, f'{infer_from_iter:06d}.tar')}"
    )

safetensors_load_model(
    sv3d,
    os.path.join(ckpt_dir, f"{infer_from_iter:06d}", "model.safetensors"),
    strict=False,
)


root_path = "/mnt/bn/pico-panwangpan-v2/projects/LGM/assets/rebuttal"
image_paths = [os.path.join(root_path, x) for x in os.listdir(root_path)]

for image_path in image_paths:
    if "_" in image_path or "gif" in image_path:
        continue
    logger.info(f"input image name is {image_path} ")
    image_path = rembg_and_center_wrapper(image_path, 512)
    continue
    image = Image.open(image_path)
    image.load()  # required for `.split()`
    if len(image.split()) == 4:  # RGBA
        input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white
        input_image.paste(image, mask=image.split()[3])  # `3` is the alpha channel
    else:
        input_image = image

    # 640 x 640
    image = (
        torch.from_numpy(np.array(input_image.resize((512 + 64, 512 + 64)))).float()
        / 255.0
    ).permute(
        2, 0, 1
    )  # (3 or 4, H, W)
    image = image.unsqueeze(0).to(dtype=torch.float32, device="cuda")
    pred_images, _ = sv3d.generate(
        image, verbose=False, do_classifier_free_guidance=False
    )

    # root path to image
    base_path_path = os.path.basename(image_path)
    tensor_to_gif(pred_images.squeeze(0), f"results/{base_path_path}.gif")
    # os.system(f"ffmpeg -i {image_path}.gif {image_path}%d.png")
