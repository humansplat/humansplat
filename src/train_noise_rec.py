import warnings

warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging

diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

import argparse
import logging
import math
import os
import time

import accelerate
import torch
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.data_loader import DataLoaderShard
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from diffusers.training_utils import EMAModel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import src.utils.util as util
import src.utils.vis_util as vis_util
from src.data import HumanDataset, TarDataset
from src.models import NoiseLGM, get_lr_scheduler, get_optimizer
from src.options import opt_dict


def main():
    PROJECT_NAME = "HumanSplat"

    parser = argparse.ArgumentParser(
        description="Train a large multi-view Gaussian model"
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
        default="data/aigc/ckpt/humanLGM",
        help="Path to the HDFS directory to save checkpoints",
    )
    parser.add_argument(
        "--wandb_token_path",
        type=str,
        default="WANDB_TOKEN",
        help="Path to the WandB login token",
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=1140,
        help="The iteration to load the checkpoint from",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the PRNG")
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking",
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training",
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=5,
        help="The max iteration step for validation",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=1000,
        help="The buffer size for webdataset shuffle",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=32,
        help="The number of processed spawned by the batch provider",
    )

    parser.add_argument(
        "--use_ema", action="store_true", help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training",
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs",
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(
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
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)
        os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir}")

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(
        os.path.join(exp_dir, "log.txt")
    )  # output to file
    file_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s - %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    )
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        kwargs_handlers=[ddp_kwargs],
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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

    # Prepare dataset
    if accelerator.is_main_process:
        if not os.path.exists(opt.urls_test):
            os.system(opt.dataset_setup_script)

    accelerator.wait_for_everyone()  # other processes wait for the main process

    # Load the training and validation dataset
    # train_dataset = GObjaverseDataset(opt, training=True)
    # val_dataset = GObjaverseDataset(opt, training=False)

    train_thuman_dataset = HumanDataset(opt, training=True, data_root=opt.thuman2_dir)
    train_2k2k_dataset = HumanDataset(opt, training=True, data_root=opt.render_2k2k_dir)
    train_tindom_dataset = HumanDataset(
        opt, training=True, data_root=opt.render_twindom_dir
    )
    val_dataset = HumanDataset(opt, training=False, data_root=opt.thuman2_dir)
    train_dataset = ConcatDataset(
        [train_thuman_dataset, train_2k2k_dataset, train_tindom_dataset]
    )

    logger.info(
        f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n"
    )

    logger.info(
        f"Load [{len(train_dataset)}] training samples and [{len(val_dataset)}] validation samples\n"
    )
    is_tar_dataset = isinstance(
        train_dataset, TarDataset
    )  # `val_dataset` is in the same class as `train_dataset`

    train_loader = DataLoader(
        train_dataset if not is_tar_dataset else train_dataset.get_webdataset(),
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.n_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True if not is_tar_dataset else None,
        collate_fn=None if not is_tar_dataset else train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset if not is_tar_dataset else val_dataset.get_webdataset(),
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.n_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=True
        if not is_tar_dataset
        else None,  # shuffle for various visualization
        collate_fn=None if not is_tar_dataset else val_dataset.collate_fn,
    )

    # Compute the effective batch size and scale learning rate
    total_batch_size = (
        configs["train"]["batch_size_per_gpu"]
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= total_batch_size / 256
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]

    # Initialize the model, optimizer and lr scheduler
    model = NoiseLGM(opt, weight_dtype)
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_optimizer(params=params_to_optimize, **configs["optimizer"])

    configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
        len(train_loader)
        // accelerator.num_processes
        / args.gradient_accumulation_steps
    )  # only account updated steps
    configs["lr_scheduler"][
        "total_steps"
    ] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"][
        "total_steps"
    ] //= accelerator.num_processes  # reset for multi-gpu

    # Load pretrained LGM if not resumed training; TODO: make the path configurable
    # if args.resume_from_iter is None:
    #     logger.info(f"Load LGM checkpoint\n")
    #     assert os.path.exists("temp/039030/model.safetensors")  # should be downloaded before running this script
    #     safetensors_load_model(model.lgm, "temp/039030/model.safetensors", strict=False)

    # Initialize the EMA model to save moving average states
    if args.use_ema:
        logger.info("Use exponential moving average (EMA) for model parameters\n")
        ema_states = EMAModel(model.parameters(), **configs["train"]["ema_kwargs"])
        ema_states.to(accelerator.device)

    # Prepare everything with `accelerator`
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader
    )
    # Set classes explicitly for everything
    model: DistributedDataParallel
    optimizer: AcceleratedOptimizer
    lr_scheduler: AcceleratedScheduler
    train_loader: DataLoaderShard
    val_loader: DataLoaderShard

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps

    # reset optimizer
    new_learning_rate = 0.0004
    assert configs["train"]["epochs"] * updated_steps_per_epoch == total_updated_steps
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: [{len(val_loader)}]")
    logger.info(f"learning rate is {new_learning_rate} \n ")

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length
    if is_tar_dataset:
        train_loader.dataset.with_epoch(len(train_loader))
        val_loader.dataset.with_epoch(len(val_loader))

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:  # load from the last checkpoint
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
                args.resume_from_iter = int(
                    sorted(dirs)[-1].split("/")[-1].split(".")[0]
                )
            else:
                dirs = os.listdir(ckpt_dir)
                if len(dirs) == 0:
                    raise ValueError(f"No checkpoint found in [{ckpt_dir}]")
                args.resume_from_iter = int(sorted(dirs)[-1])

        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        if args.hdfs_dir is not None and accelerator.is_main_process:
            os.system(
                f"hdfs dfs -get {os.path.join(args.hdfs_dir, f'{args.resume_from_iter:06d}.tar')} {ckpt_dir} && "
                + f"tar -xf {os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}.tar')} -C {ckpt_dir} && "
                + f"rm {os.path.join(ckpt_dir, f'{args.resume_from_iter:06d}.tar')}"
            )
        accelerator.wait_for_everyone()  # wait before preparing checkpoints by the main process
        accelerator.load_state(
            os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"), strict=False
        )  # `LPIPS` parameters are not saved
        if args.use_ema:
            ema_states.load_state_dict(
                torch.load(
                    os.path.join(
                        ckpt_dir, f"{args.resume_from_iter:06d}", "ema_states.pth"
                    ),
                    map_location=accelerator.device,
                )
            )
        # global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = util.save_experiment_params(args, configs, opt, exp_dir)
        util.save_model_architecture(accelerator.unwrap_model(model), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        os.environ["WANDB_API_KEY_FILE"] = args.wandb_token_path

        wandb.init(
            project=PROJECT_NAME,
            name=args.tag,
            config=exp_params,
            dir=exp_dir,
            resume=True,
        )

        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(
            os.path.join(exp_dir, "log.txt")
        )  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    # Start training
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=125,
        disable=not accelerator.is_main_process,
    )

    optimizer.param_groups[0]["lr"] = new_learning_rate
    lr_scheduler.step(new_learning_rate)

    while True:

        active_train_loader = train_loader
        for batch in active_train_loader:

            if global_update_step == args.max_train_steps:
                progress_bar.close()
                logger.logger.propagate = True  # propagate to the root logger (console)
                if accelerator.is_main_process:
                    wandb.finish()
                logger.info("Training finished!\n")
                return

            model.train()

            with accelerator.accumulate(model):
                outputs = model(batch, weight_dtype)

                psnr = outputs["psnr"]
                lpips = outputs["lpips"]
                loss = outputs["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                psnr = accelerator.gather(psnr.detach()).mean()
                lpips = accelerator.gather(lpips.detach()).mean()
                loss = accelerator.gather(loss.detach()).mean()

                logs = {
                    "psnr": psnr.item(),
                    "lpips": lpips.item(),
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if args.use_ema:
                    ema_states.step(model.parameters())
                    logs.update({"ema_decay": ema_states.cur_decay_value})

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)
                global_update_step += 1

                logger.info(
                    f"[{global_update_step:06d} / {total_updated_steps:06d}] "
                    + f"psnr: {logs['psnr']:.4f}, lpips: {logs['lpips']:.4f}, "
                    + f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}"
                    + f", ema_decay: {logs['ema_decay']:.4f}"
                    if args.use_ema
                    else ""
                )

                # Log the training progress
                if (
                    global_update_step % configs["train"]["log_freq"] == 0
                    or global_update_step == 1
                    or global_update_step % updated_steps_per_epoch == 0
                ):  # last step of an epoch
                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "training/psnr": psnr.item(),
                                "training/lpips": lpips.item(),
                                "training/loss": loss.item(),
                                "training/lr": lr_scheduler.get_last_lr()[0],
                            },
                            step=global_update_step,
                        )
                        if args.use_ema:
                            wandb.log(
                                {"training/ema_decay": ema_states.cur_decay_value},
                                step=global_update_step,
                            )

                # Save checkpoint
                if (
                    global_update_step % configs["train"]["save_freq"] == 0
                    or global_update_step % updated_steps_per_epoch == 0
                ):  # last step of an epoch
                    if accelerator.is_main_process:
                        accelerator.save_state(
                            os.path.join(ckpt_dir, f"{global_update_step:06d}")
                        )
                        if args.use_ema:
                            torch.save(
                                ema_states.state_dict(),
                                os.path.join(
                                    ckpt_dir,
                                    f"{global_update_step:06d}",
                                    "ema_states.pth",
                                ),
                            )
                        if args.hdfs_dir is not None:
                            os.system(
                                f"tar -cf {os.path.join(ckpt_dir, f'{global_update_step:06d}.tar')} -C {ckpt_dir} {global_update_step:06d} && "
                                + f"hdfs dfs -put -f {os.path.join(ckpt_dir, f'{global_update_step:06d}.tar')} {args.hdfs_dir} && "
                                + f"rm -rf {os.path.join(ckpt_dir, f'{global_update_step:06d}.tar')} {os.path.join(ckpt_dir, f'{global_update_step:06d}')}"
                            )

                # Evaluate on the validation set
                if (
                    (
                        global_update_step % configs["train"]["early_eval_freq"] == 0
                        and global_update_step  # eval more frequently at the beginning
                        < configs["train"]["eval_freq"]
                    )
                    or global_update_step % configs["train"]["eval_freq"] == 0
                    or global_update_step == 1
                    or global_update_step % updated_steps_per_epoch == 0
                ):  # last step of an epoch

                    # Use EMA parameters for evaluation
                    if args.use_ema:
                        ema_states.store(model.parameters())
                        ema_states.copy_to(model.parameters())

                    with torch.no_grad():
                        model.eval()

                        all_val_matrics, val_steps = {}, 0
                        val_progress_bar = tqdm(
                            range(len(val_loader))
                            if args.max_val_steps is None
                            else range(args.max_val_steps),
                            desc="Validation",
                            ncols=125,
                            disable=not accelerator.is_main_process,
                        )
                        for val_batch in val_loader:
                            val_outputs = model(val_batch, weight_dtype)

                            val_psnr = val_outputs["psnr"]
                            val_lpips = val_outputs["lpips"]
                            val_loss = val_outputs["loss"]

                            val_psnr = accelerator.gather_for_metrics(val_psnr).mean()
                            val_lpips = accelerator.gather_for_metrics(val_lpips).mean()
                            val_loss = accelerator.gather_for_metrics(val_loss).mean()

                            val_logs = {
                                "psnr": val_psnr.item(),
                                "lpips": val_lpips.item(),
                                "loss": val_loss.item(),
                            }
                            val_progress_bar.set_postfix(**val_logs)
                            val_progress_bar.update(1)
                            val_steps += 1

                            all_val_matrics.setdefault("psnr", []).append(val_psnr)
                            all_val_matrics.setdefault("lpips", []).append(val_lpips)
                            all_val_matrics.setdefault("loss", []).append(val_loss)

                            if (
                                args.max_val_steps is not None
                                and val_steps == args.max_val_steps
                            ):
                                break

                    val_progress_bar.close()

                    # Restore the running model parameters
                    if args.use_ema:
                        ema_states.restore(model.parameters())

                    for k, v in all_val_matrics.items():
                        all_val_matrics[k] = torch.tensor(v).mean()

                    logger.info(
                        f"Eval [{global_update_step:06d} / {total_updated_steps:06d}] "
                        + f"psnr: {all_val_matrics['psnr'].item():.4f}, "
                        + f"lpips: {all_val_matrics['lpips'].item():.4f}, "
                        + f"loss: {all_val_matrics['loss'].item():.4f}\n"
                    )

                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "validation/psnr": all_val_matrics["psnr"].item(),
                                "validation/lpips": all_val_matrics["lpips"].item(),
                                "validation/loss": all_val_matrics["loss"].item(),
                            },
                            step=global_update_step,
                        )

                        # Visualize rendering
                        wandb.log(
                            {"images/training": vis_util.wandb_image_log(outputs)},
                            step=global_update_step,
                        )
                        wandb.log(
                            {
                                "images/validation": vis_util.wandb_image_log(
                                    val_outputs
                                )
                            },
                            step=global_update_step,
                        )


if __name__ == "__main__":
    main()
