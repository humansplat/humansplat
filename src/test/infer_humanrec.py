import torch
from torch.utils.data import DataLoader
from src.data import HumanDataset, TarDataset
from src.models import HumanRec
from src.options import opt_dict


@torch.no_grad()
def main():
    opt = opt_dict["humansplat"]
    model = HumanRec(opt).to(dtype=torch.float32, device="cuda")
    model.eval()
    weight_dtype = torch.float32

    # Load the training and validation dataset
    val_dataset = HumanDataset(opt, training=False, data_root=opt.thuman2_dir)

    # Create the data loader
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
        print(f" {idx}/{len(val_loader)} -- ")
        outputs = model(batch, weight_dtype, func_name="compute_loss")
        print(outputs["psnr"], outputs["ssim"])

    

if __name__ == "__main__":
    main()
