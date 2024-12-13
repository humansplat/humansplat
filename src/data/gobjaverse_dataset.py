import io
import json
import random
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import webdataset as wds
from kiui.cam import undo_orbit_camera
from torch import Tensor

from src.options import Options
from src.utils.op_util import (grid_distortion, normalize_cam_poses_4x4,
                               orbit_camera_jitter)

from .tar_dataset import TarDataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GObjaverseDataset(TarDataset):
    def __init__(self, opt: Options, training=True):
        self.opt = opt
        self.training = training

        if self.training:
            urls = self.opt.urls_train
        else:
            urls = self.opt.urls_test

        # Default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (
            self.opt.zfar - self.opt.znear
        )
        self.proj_matrix[3, 2] = -(self.opt.zfar * self.opt.znear) / (
            self.opt.zfar - self.opt.znear
        )
        self.proj_matrix[2, 3] = 1

        ################################ Setup `webdataset` ################################

        self.dataset_size = int(
            urls.split("-")[-2]
        )  # dataset size is saved in the file name

        self.dataset = wds.WebDataset(
            urls, resampled=True
        )  # `resampled=True` for multi-node training

        if opt.shuffle_buffer_size > 0:
            self.dataset = self.dataset.shuffle(opt.shuffle_buffer_size)
        self.shuffle_buffer_size = opt.shuffle_buffer_size

        self.dataset: wds.WebDataset = self.dataset.map(self.decoder)

        self.dataset.with_length(self.dataset_size)  # enable `__len__()`
        # self.dataset.with_epoch(...)  # set steps per epoch, as a resampled dataset is infinite size

    def decoder(
        self, sample: Dict[str, Union[str, bytes]]
    ) -> Dict[str, Tensor]:  # only `sample["__key__"]` is in str type
        outputs = {}

        V_in = self.opt.num_input_views
        V = self.opt.num_views  # V_in + V_out

        if not self.opt.load_sorted_images:
            # Randomly sample views (some views may not appear in the dataset)
            random_idxs, shuffled_idxs = [], np.random.permutation(
                40
            )  # `40` is hard-coded for GObjaverse dataset
            for i in range(40):
                if f"{shuffled_idxs[i]:05d}.png" in sample and shuffled_idxs[i] not in [
                    25,
                    26,
                ]:  # filter top and down views
                    random_idxs.append(shuffled_idxs[i])
                if len(random_idxs) == V:
                    break
            # Randomly repeat views if not enough views
            while len(random_idxs) < V:
                random_idxs.append(np.random.choice(random_idxs))

        else:  # for video models
            # Randomly sample sorted views (some views may not appear in the dataset)
            random_idxs = []
            for i in range(24):  # hard-coded for GObjaverse dataset
                if f"{i:05d}.png" in sample:  # filter top and down views
                    random_idxs.append(i)
            # # TODO: hard-code for GObjaverse
            # assert V == 24
            # assert len(random_idxs) == 24
            # Randomly repeat views if not enough views
            while len(random_idxs) < V:
                random_idxs.append(np.random.choice(random_idxs))
            random_idxs.sort()
            # Randomly shift the start index
            start_idx = random.randint(0, len(random_idxs) - 1)
            random_idxs = random_idxs[start_idx:] + random_idxs[:start_idx]
            # Randomly choose `V` views
            choice = np.random.choice(len(random_idxs), V, replace=False)
            choice.sort()
            random_idxs = [random_idxs[i] for i in choice]

        images, masks, cam_poses, cam_poses_4x4 = [], [], [], []

        for vid in random_idxs:
            image = self._load_image(sample[f"{vid:05d}.png"])  # (4, 512, 512)
            mask = image[3:4]  # (1, 512, 512)
            image = image[:3] * mask + (1.0 - mask)  # (3, 512, 512), to white bg

            c2w = self._load_camera_from_json(sample[f"{vid:05d}.json"])
            # Blender world + OpenCV cam -> OpenGL world & cam; https://kit.kiui.moe/camera
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1  # invert up and forward direction

            cam_pose = self._get_pose(c2w)  # [4]

            images.append(image)
            masks.append(mask)
            cam_poses.append(cam_pose)
            cam_poses_4x4.append(c2w)

        images = torch.stack(images, dim=0)  # (V, 3, H, W)
        masks = torch.stack(masks, dim=0)  # (V, 1, H, W)
        cam_poses = torch.stack(cam_poses, dim=0)  # (V, 4)
        cam_poses_4x4 = torch.stack(cam_poses_4x4, dim=0)  # (V, 4, 4)

        # Normalize 4x4 camera poses (transform the first pose to a fixed position)
        cam_poses_4x4 = normalize_cam_poses_4x4(
            cam_poses_4x4, i=0, norm_radius=self.opt.cam_radius
        )

        images_input = F.interpolate(
            images[:V_in].clone(),
            size=(self.opt.input_size, self.opt.input_size),
            mode="bilinear",
            align_corners=False,
        )  # (V, C, H, W)

        cam_poses_input = cam_poses[:V_in].clone()
        cam_poses_4x4_input = cam_poses_4x4[:V_in].clone()

        # Data augmentation
        if self.training and V_in > 1:
            # Apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # Apply camera jittering
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_4x4_input[1:] = orbit_camera_jitter(cam_poses_4x4_input[1:])

        images_input = (
            images_input * 2.0 - 1.0
            if not self.opt.imagenet_stats_norm
            else TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        )

        outputs["images_input"] = images_input  # (V, 3, H, W)
        outputs["cam_poses_input"] = cam_poses_input  # (V, 4)
        outputs["cam_poses_4x4_input"] = cam_poses_4x4_input  # (V, 4, 4)

        outputs["images_output"] = F.interpolate(
            images if not self.opt.exclude_input_views else images[V_in:],
            size=(self.opt.output_size, self.opt.output_size),
            mode="bilinear",
            align_corners=False,
        )  # (V, 3, H', W')
        outputs["masks_output"] = F.interpolate(
            masks if not self.opt.exclude_input_views else masks[V_in:],
            size=(self.opt.output_size, self.opt.output_size),
            mode="bilinear",
            align_corners=False,
        )  # (V, 1, H', W')
        outputs["cam_poses_output"] = (
            cam_poses if not self.opt.exclude_input_views else cam_poses[V_in:]
        )  # (V, 4)
        outputs["cam_poses_4x4_output"] = (
            cam_poses_4x4 if not self.opt.exclude_input_views else cam_poses_4x4[V_in:]
        )  # (V, 4, 4)

        # OpenGL to COLMAP camera for Gaussian renderer
        cam_poses_4x4_colmap = cam_poses_4x4.clone()
        cam_poses_4x4_colmap[:, :3, 1:3] *= -1  # invert up & forward direction

        # Camera values needed by Gaussian rasterizer
        cam_view = torch.inverse(cam_poses_4x4_colmap).transpose(1, 2)  # (V, 4, 4)
        cam_view_proj = cam_view @ self.proj_matrix  # (V, 4, 4)
        cam_pos = -cam_poses_4x4_colmap[
            :, :3, 3
        ]  # (V, 3); https://github.com/3DTopia/LGM/issues/15

        outputs["cam_view"] = (
            cam_view if not self.opt.exclude_input_views else cam_view[V_in:]
        )
        outputs["cam_view_proj"] = (
            cam_view_proj if not self.opt.exclude_input_views else cam_view_proj[V_in:]
        )
        outputs["cam_pos"] = (
            cam_pos if not self.opt.exclude_input_views else cam_pos[V_in:]
        )

        return outputs

    def _load_image(self, image_bytes: bytes) -> Tensor:
        image_plt = plt.imread(io.BytesIO(image_bytes))  # (H, W, 4) in [0, 1]
        return (
            torch.from_numpy(image_plt).permute(2, 0, 1).float()
        )  # (4, H, W) in [0, 1]

    def _load_camera_from_json(self, json_bytes: bytes) -> Tensor:
        json_dict = json.loads(json_bytes)

        # In OpenCV convention
        c2w = np.eye(4)
        c2w[:3, 0] = np.array(json_dict["x"])
        c2w[:3, 1] = np.array(json_dict["y"])
        c2w[:3, 2] = np.array(json_dict["z"])
        c2w[:3, 3] = np.array(json_dict["origin"])
        return torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

    def _get_pose(self, c2w: Tensor) -> Tensor:
        # In OpenGL world & cam
        theta, azimuth, d = undo_orbit_camera(
            c2w, is_degree=True
        )  # azimuth: [-180, 180]: +z (0) to +x (+90)
        theta = theta + 90.0  # from [-90, 90] to [0, 180]: +y to -y

        if d.item() < 1.4 or d.item() > 2.0:  # RichDreamer Sec. 3.1
            raise ValueError(f"Distance [{d.item()}] is out of range (1.4, 2.0)")
            # d = np.clip(d.item(), 1.4, 2.)
        # Log scale for radius
        target_T = torch.tensor(
            [
                np.deg2rad(theta),
                np.deg2rad(azimuth),
                (np.log(d) - np.log(1.4)) / (np.log(2.0) - np.log(1.4)) * torch.pi,
                torch.tensor(0),
            ]
        )
        assert torch.all(target_T <= torch.pi + 1e-3) and torch.all(
            target_T >= -torch.pi - 1e-3
        )
        return target_T
