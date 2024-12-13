import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data import HumanDataset, TarDataset
from src.models import HumanSplat
from src.options import opt_dict


@torch.no_grad()
def main():
    path = "assets/images/1.jpeg"
    image = Image.open(path)
    image.load()  # required for `.split()`
    if len(image.split()) == 4:  # RGBA
        input_image = Image.new("RGB", image.size, (255, 255, 255))  # pure white
        input_image.paste(image, mask=image.split()[3])  # `3` is the alpha channel
    else:
        input_image = image
    image = (
        torch.from_numpy(np.array(input_image.resize((576, 576)))).float() / 255.0
    ).permute(
        2, 0, 1
    )  # (3 or 4, H, W)
    image = image.unsqueeze(0).to(dtype=torch.float32, device="cuda")

    opt = opt_dict["humansplat"]
    model = HumanSplat(opt).to(dtype=torch.float32, device="cuda")
    model.eval()
    weight_dtype = torch.float32

    # pred_images, _ = model.evaluate(image, verbose=True)
    # tensor_to_gif(pred_images.squeeze(0), "temp.gif")
    # Load the training and validation dataset

    val_dataset = HumanDataset(opt, training=False, data_root=opt.thuman2_dir)
    is_tar_dataset = isinstance(
        val_dataset, TarDataset
    )  # `val_dataset` is in the same class as `train_dataset`
    val_loader = DataLoader(
        val_dataset if not is_tar_dataset else val_dataset.get_webdataset(),
        batch_size=1,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
        shuffle=False,  # shuffle for various visualization
    )

    for idx, batch in enumerate(val_loader):
        print(f" {idx}/{len(val_loader)}")
        outputs = model(batch, weight_dtype, func_name="evaluate")
        print(outputs["psnr"], outputs["ssim"])
        exit()


if __name__ == "__main__":
    main()
