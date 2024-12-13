import warnings

warnings.filterwarnings("ignore")  # ignore all warnings


import argparse
import logging
import os
import time

import accelerate
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from kiui.cam import orbit_camera
from safetensors.torch import load_model as safetensors_load_model
from torchvision.transforms import functional as TF

import src.utils.util as util
from extensions.mvdream import MVDreamPipeline
from src.models import LGM
from src.options import opt_dict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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


def main():
    parser = argparse.ArgumentParser(
        description="Infer a large multi-view Gaussian model"
    )

    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/tiger/LGM/out",
        help="Path to the output directory",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default="data/aigc/ckpt/gen3d",
        help="Path to the HDFS directory to save checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the PRNG")

    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--half_precision", action="store_true", help="Use half precision for inference"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image for reconstruction",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the directory of images for reconstruction",
    )
    parser.add_argument(
        "--infer_from_iter",
        type=int,
        default=-1,
        help="The iteration to load the checkpoint from",
    )
    parser.add_argument(
        "--rembg_and_center",
        action="store_true",
        help="Whether or not to remove background and center the image",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        help="Elevation degree for the input image",
    )
    parser.add_argument(
        "--save_mvimages",
        action="store_true",
        help="Whether or not to save generated multi-view images",
    )
    parser.add_argument(
        "--fancy_video",
        action="store_true",
        help="Whether or not to output a fancy video",
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.load_configs(
        args.config_file, extras
    )  # change yaml configs by `extras`

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = (
            time.strftime("%Y-%m-%d_%H:%M")
            + "_"
            + os.path.split(args.config_file)[-1].split()[0]
        )  # config file name

    # Create the experiment directory
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    infer_dir = os.path.join(exp_dir, "inference")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(
        os.path.join(args.output_dir, args.tag, "log_infer.txt")
    )  # output to file
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    )
    logger.addHandler(file_handler)
    logger.propagate = True  # propagate to the root logger (console)

    # Set the random seed
    if args.seed is not None:
        accelerate.utils.set_seed(args.seed)
        logger.info(
            f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n"
        )

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    # Load the image for reconstruction
    assert args.image_path is not None or args.image_dir is not None
    if args.image_dir is not None:
        logger.info(f"Load images from [{args.image_dir}]\n")
        image_paths = [
            os.path.join(args.image_dir, filename)
            for filename in os.listdir(args.image_dir)
            if filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".webp")
        ]
    else:
        logger.info(f"Load image from [{args.image_path}]\n")
        image_paths = [args.image_path]

    # Initialize the model
    model = LGM(opt)
    model = model.to(f"cuda:{args.gpu_id}")

    # Load pretrained multi-view image generative model
    mvpipe = MVDreamPipeline.from_pretrained(
        "ashawkey/imagedream-ipmv-diffusers",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    mvpipe = mvpipe.to(f"cuda:{args.gpu_id}")

    # Load checkpoint
    if args.infer_from_iter < 0:  # load from the last checkpoint
        if args.hdfs_dir is not None:
            dirs = [
                line.split()[-1]
                for line in os.popen(f"hdfs dfs -ls {args.hdfs_dir}")
                .read()
                .strip()
                .split("\n")[1:]
            ]
            if len(dirs) == 0:
                raise ValueError(f"No checkpoint found in [{args.hdfs_dir}]")
            args.infer_from_iter = int(sorted(dirs)[-1].split("/")[-1].split(".")[0])
        else:
            dirs = os.listdir(ckpt_dir)
            if len(dirs) == 0:
                raise ValueError(f"No checkpoint found in [{ckpt_dir}]")
            args.infer_from_iter = int(sorted(dirs)[-1])

    logger.info(f"Load checkpoint from iteration [{args.infer_from_iter}]\n")
    if args.hdfs_dir is not None and not os.path.exists(
        os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}", "model.safetensors")
    ):
        os.system(
            f"hdfs dfs -get {os.path.join(args.hdfs_dir, f'{args.infer_from_iter:06d}.tar')} {ckpt_dir} && "
            + f"tar -xf {os.path.join(ckpt_dir, f'{args.infer_from_iter:06d}.tar')} -C {ckpt_dir} && "
            + f"rm {os.path.join(ckpt_dir, f'{args.infer_from_iter:06d}.tar')}"
        )
    safetensors_load_model(
        model,
        os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}", "model.safetensors"),
        strict=False,
    )  # `LPIPS` parameters are not saved

    if args.half_precision:
        model = model.half()

    # Save all experimental parameters of this run to a file (args and configs)
    _ = util.save_experiment_params(args, configs, opt, infer_dir)

    # Inference
    model.eval()
    for i in range(
        len(image_paths)
    ):  # to save outputs with the same name as the input image
        image_path = image_paths[i]
        # (Optional) Remove background and center the image
        if args.rembg_and_center:
            image_path = rembg_and_center_wrapper(image_path, opt.input_size)

        name = f"{image_path.split('/')[-1].split('.')[0]}_{args.infer_from_iter:06d}"

        image = plt.imread(image_path)
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (
                1.0 - image[..., 3:4]
            )  # RGBA to RGB white background

        # Generate multi-view images
        if opt.num_input_views == 4:
            mv_images = mvpipe(
                "",
                image,
                guidance_scale=5.0,
                num_inference_steps=30,
                elevation=args.elevation,
            )
            mv_images = np.stack(
                [mv_images[1], mv_images[2], mv_images[3], mv_images[0]], axis=0
            )  # (4, H, W, 3)
            if args.save_mvimages:
                for vid in range(len(mv_images)):
                    plt.imsave(
                        os.path.join(infer_dir, f"{name}_mv{vid}.png"), mv_images[vid]
                    )
        elif opt.num_input_views == 1:
            mv_images = image[np.newaxis, ...]  # (1, H, W, 3)
        else:
            raise NotImplementedError(
                f"Invalid `opt.num_input_views`: [{opt.num_input_views}]"
            )

        # Generate Gaussians
        input_images = (
            torch.from_numpy(mv_images)
            .permute(0, 3, 1, 2)
            .float()
            .to(f"cuda:{args.gpu_id}")
        )  # (4 or 1, 3, H, W)
        input_images = TF.normalize(
            input_images, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        )
        rays_embeddings = model.prepare_default_rays(elevation=args.elevation).to(
            f"cuda:{args.gpu_id}"
        )  # (4 or 1, 6, H, W)
        input_images = torch.cat([input_images, rays_embeddings], dim=1).unsqueeze(
            0
        )  # (1, 4 or 1, 9, H, W)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16 if args.half_precision else torch.float32,
            ):
                gaussians = model.forward_gaussians(input_images)
            # Save Gaussians
            model.gs.save_ply(gaussians, os.path.join(infer_dir, f"{name}.ply"))

            # Render 360 degree video
            tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
            proj_matrix = torch.zeros(
                4, 4, dtype=torch.float32, device=f"cuda:{args.gpu_id}"
            )
            proj_matrix[0, 0] = 1 / tan_half_fov
            proj_matrix[1, 1] = 1 / tan_half_fov
            proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
            proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
            proj_matrix[2, 3] = 1

            images = []

            if args.fancy_video:
                azimuth = np.arange(0, 720, 4, dtype=np.int32)
                for azi in azimuth:
                    cam_poses = (
                        torch.from_numpy(
                            orbit_camera(
                                args.elevation, azi, radius=opt.cam_radius, opengl=True
                            )
                        )
                        .unsqueeze(0)
                        .to(f"cuda:{args.gpu_id}")
                    )
                    cam_poses[:, :3, 1:3] *= -1  # OpenGL -> OpenCV

                    # Camera values needed by Gaussian rasterizer
                    cam_view = torch.inverse(cam_poses).transpose(1, 2)  # (V, 4, 4)
                    cam_view_proj = cam_view @ proj_matrix  # (V, 4, 4)
                    cam_pos = -cam_poses[:, :3, 3]  # (V, 3)

                    image = (
                        model.gs.render(
                            gaussians,
                            cam_view.unsqueeze(0),
                            cam_view_proj.unsqueeze(0),
                            cam_pos.unsqueeze(0),
                            scale_modifier=min(azi / 360, 1),
                        )["image"]
                        .squeeze(1)
                        .permute(0, 2, 3, 1)
                    )  # (1, H, W, 3)
                    images.append(
                        (image.contiguous().float().cpu().numpy() * 255).astype(
                            np.uint8
                        )
                    )

            else:
                azimuth = np.arange(0, 360, 2, dtype=np.int32)
                for azi in azimuth:
                    cam_poses = (
                        torch.from_numpy(
                            orbit_camera(
                                args.elevation, azi, radius=opt.cam_radius, opengl=True
                            )
                        )
                        .unsqueeze(0)
                        .to(f"cuda:{args.gpu_id}")
                    )
                    cam_poses[:, :3, 1:3] *= -1  # OpenGL -> OpenCV

                    # Camera values needed by Gaussian rasterizer
                    cam_view = torch.inverse(cam_poses).transpose(1, 2)  # (V, 4, 4)
                    cam_view_proj = cam_view @ proj_matrix  # (V, 4, 4)
                    cam_pos = -cam_poses[:, :3, 3]  # (V, 3)

                    image = (
                        model.gs.render(
                            gaussians,
                            cam_view.unsqueeze(0),
                            cam_view_proj.unsqueeze(0),
                            cam_pos.unsqueeze(0),
                            scale_modifier=1,
                        )["image"]
                        .squeeze(1)
                        .permute(0, 2, 3, 1)
                    )  # (1, H, W, 3)
                    images.append(
                        (image.contiguous().float().cpu().numpy() * 255).astype(
                            np.uint8
                        )
                    )

            images = np.concatenate(images, axis=0)  # (V, H, W, 3)
            imageio.mimwrite(os.path.join(infer_dir, f"{name}.mp4"), images, fps=30)

    logger.info("Inference finished!\n")


if __name__ == "__main__":
    main()
