from typing import *

import numpy as np
import wandb
from diffusers.utils.export_utils import export_to_gif
from numpy import ndarray
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from wandb import Image as WandbImage


def tensor_to_gif(tensor: Tensor, save_path: str):
    assert tensor.ndim == 4  # (V, C, H, W)
    assert tensor.shape[1] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    images = (tensor.permute(0, 2, 3, 1).cpu().float().numpy() * 255).astype(
        np.uint8
    )  # (V, H, W, C)
    images = [Image.fromarray(image) for image in images]
    export_to_gif(images, save_path)


def vis_batched_mvimages(mvimages: Tensor, save_path: Optional[str] = None) -> ndarray:
    assert torch.all(mvimages >= 0.0) and torch.all(mvimages <= 1.0)
    assert mvimages.ndim == 5  # (B, V, C, H, W)
    assert mvimages.shape[2] == 3  # RGB

    mvimages = mvimages.float().detach().cpu().numpy()
    mvimages = einops.rearrange(mvimages, "b v c h w -> (b h) (v w) c")

    if save_path is not None:
        kiui.write_image(save_path, mvimages)

    return mvimages


def wandb_image_log(
    outputs: Dict[str, Tensor], max_num=4, max_view=8
) -> List[WandbImage]:
    formatted_images = []
    for k in outputs.keys():
        if "images" in k:  # (B, V, 3, H, W)
            for b in range(min(max_num, len(outputs[k]))):
                for v in range(min(max_view, outputs[k][b].shape[0])):
                    formatted_images.append(
                        wandb.Image(
                            tensor_to_image(outputs[k][b][v].detach()),
                            caption=f"{k}[{b:02d}]-view{v:02d}",
                        )
                    )

    return formatted_images


def tensor_to_image(tensor: Tensor, return_pil=False) -> Union[ndarray, PILImage]:
    assert tensor.ndim == 3  # (C, H, W)
    assert tensor.shape[0] in [1, 3]  # grayscale, RGB (not consider RGBA here)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    image = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if return_pil:
        image = Image.fromarray(image)
    return image
