# OpenGL/MAYA       OpenCV/Colmap     Blender      Unity/DirectX     Unreal
# Right-handed      Right-handed    Right-handed    Left-handed    Left-handed
#      +y                +z           +z  +y         +y  +z          +z
#      |                /             |  /           |  /             |
#      |               /              | /            | /              |
#      |______+x      /______+x       |/_____+x      |/_____+x        |______+x
#     /               |                                              /
#    /                |                                             /
#   /                 |                                            /
#  +z                 +y                                          +y

import json
import os
import os.path as osp
import pickle
import random
from typing import *
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from smplx import SMPL
from torch import Tensor
from torch.utils.data import Dataset
from src.utils.projection import Projector
IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

from kiui.cam import orbit_camera
from src.options import Options
from src.utils.op_util import grid_distortion, orbit_camera_jitter


class HumanDataset(Dataset):
    def __init__(
        self, opt: Options, training=True, data_root="", image_zoom_ratio=None
    ):
        self.opt = opt
        self.training = training

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
        self.human_names = [
            osp.join(data_root, x) for x in sorted(os.listdir(data_root))
        ]

        self.cam_path = "resulted_renders_dense/p3d_render_data.pkl"

        # 500 / 525
        train_num = int(len(self.human_names) * 0.95) + 1

        if hasattr(self, "hack_name"):
            hack_names = ["0521", "0520"]
            self.human_names = [osp.join(data_root, x) for x in hack_names]
        else:
            self.human_names = [osp.join(data_root, x) for x in os.listdir(data_root)]
            train_num = int(len(self.human_names) * 0.95) + 1

            if self.training:
                self.human_names = self.human_names[:train_num]
            else:
                self.human_names = self.human_names[train_num:]

        with open("data/smpl_segmentation.json", "r") as file:
            self.smpl_segmentation = json.load(file)

        # load the SMPL model
        # device = torch.device("cuda")
        # self.smpl_model = SMPL("assets/smpl_model", gender="male", device=device)

        # project feature to Canonical view
        self.projector = Projector(device)

    def __len__(self):
        return len(self.human_names)

    @torch.no_grad()
    def __getitem__(self, idx) -> Dict[str, Tensor]:
        outputs = {}
        human_name = self.human_names[idx]

        if self.opt.random_load:
            start, end = 0, 35  # TODO: make `start and end` configurable
            # Randomly sample views (some views may not appear in the dataset)
            random_idxs, shuffled_idxs = (
                [],
                np.random.permutation(end - start + 1) + start,
            )
            for i in range(36):
                random_idxs.append(shuffled_idxs[i])
                if len(random_idxs) == self.opt.num_views:
                    break
            # Randomly repeat views if not enough views
            while len(random_idxs) < self.opt.num_views:
                random_idxs.append(np.random.choice(random_idxs))
        else:
            # Randomly sample sorted views (some views may not appear in the dataset)
            images_per_object = 36
            random_idxs = list(range(images_per_object))
            random_idxs.sort()
            start_idx = random.randint(0, images_per_object - 1)
            random_idxs = random_idxs[start_idx:] + random_idxs[:start_idx]
            # Randomly choose `V` views
            interval = images_per_object // self.opt.num_views
            random_idxs = random_idxs[0::interval]

        images, masks, cam_poses, cam_poses_4x4 = [], [], [], []
        origin_c2w = []
        base_name = os.path.basename(human_name)
        pkl_file_path = osp.join(human_name, self.cam_path)

        with open(f"{human_name}/{base_name}_smpl.pkl", "rb") as f:
            smpl_data = pickle.load(f)

        betas = torch.from_numpy(smpl_data["betas"]).unsqueeze(0)
        pose_params = torch.from_numpy(smpl_data["pose"][3:]).unsqueeze(0)
        transl = torch.from_numpy(smpl_data["trans"]).unsqueeze(0)
        global_orient = torch.from_numpy(smpl_data["pose"][:3]).unsqueeze(0)

        # smpl_xyz = self.smpl_model(
        #     betas=betas,
        #     body_pose=pose_params,
        #     global_orient=global_orient,
        #     transl=transl,
        # ).vertices.detach()[0]

        # xyz reverse
        # smpl_xyz[:, 0] = -1 * smpl_xyz[:, 0]

        cam_infos = self._load_camera_from_pkl(pkl_file_path)
        cam_num = len(cam_infos["R"])
        camera_poses = np.repeat(np.expand_dims(np.eye(4), 0), cam_num, axis=0)
        camera_poses[:, :3, 3] = cam_infos["T"]
        camera_poses[:, :3, :3] = cam_infos["R"]
        camera_poses = np.linalg.inv(camera_poses)

        cam_poses_4x4 = torch.zeros((self.opt.num_views, 4, 4))
        azims = np.linspace(-180, 180, cam_num)
        cam_zim = []

        for vid in random_idxs:
            if "Thuman2" in human_name:
                image_path = osp.join(
                    human_name, f"{base_name}_for_gs", "images", f"{vid:04d}.png"
                )
                mask_path = osp.join(
                    human_name, f"{base_name}_for_gs", "masks", f"{vid:04d}.png"
                )
                image = self._load_image(image_path)  # (3, 512, 512)
                mask = self._load_mask(mask_path)  # (1, 512, 512)
            else:
                image_path = osp.join(
                    human_name, "resulted_renders_dense", f"{vid:03d}.jpg"
                )
                image = self._load_image(image_path)
                mask = torch.all(image < 200 / 255.0, dim=0).unsqueeze(0).float()

            cam_pos = self._get_pose(torch.Tensor(np.linalg.inv(camera_poses[vid])))
            images.append(image)
            masks.append(mask)
            cam_poses.append(cam_pos)
            cam_zim.append(azims[vid])

        # Relative azimuth w.r.t. the first view
        init_azi = None
        for idx, azi_item in enumerate(cam_zim):
            ele, azi, dis = (
                0,
                azi_item,
                2,
            )  # elevation: [-90, 90] from +y(-90) to -y(90)
            if init_azi is None:
                init_azi = azi
            azi = (azi - init_azi) % 360.0  # azimuth: [0, 360] from +z(0) to +x(90)
            ele_sign = ele >= 0
            ele = abs(ele) - 1e-8
            ele = ele * (1.0 if ele_sign else -1.0)
            cam_poses_4x4[idx] = torch.from_numpy(orbit_camera(ele, azi, dis)).float()



        # visualizer_cam_pose(debug_pose.cpu().numpy(), "out/human_cam.png")
        # OpenGL to COLMAP camera for Gaussian renderer
        cam_poses_4x4[:, :3, 1:3] *= -1

        # Whether scale the object w.r.t. the first view to a fixed size
        if self.opt.norm_camera:
            scale = self.opt.norm_radius / (torch.norm(cam_poses_4x4[0, :3, 3], dim=-1))
        else:
            scale = 1.0

        cam_poses_4x4[:, :3, 3] *= scale

        # outputs["smpl_xyz"] = smpl_xyz
        outputs["smpl_semantic"] = self.smpl_segmentation

        images = torch.stack(images, dim=0)  # (V, 3, H, W)
        masks = torch.stack(masks, dim=0)  # (V, 1, H, W)
        cam_poses = torch.stack(cam_poses, dim=0)  # (V, 4)

        images_input = F.interpolate(
            images[: self.opt.num_input_views].clone(),
            size=(self.opt.input_size, self.opt.input_size),
            mode="bilinear",
            align_corners=False,
        )  # (V, C, H, W)

        cam_poses_input = cam_poses[: self.opt.num_input_views].clone()
        cam_poses_4x4_input = cam_poses_4x4[: self.opt.num_input_views].clone()

        # Data augmentation
        if self.training and self.opt.num_input_views > 1:
            # Apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])

            # Apply camera jittering
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_4x4_input[1:] = orbit_camera_jitter(cam_poses_4x4_input[1:])

        images_input = TF.normalize(
            images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        )

        # todo : hard code to 1024
        scale_factor = self.opt.input_size / 1024
        cam_infos["intrinsic"][0, 0] = cam_infos["intrinsic"][0, 0] * scale_factor
        cam_infos["intrinsic"][0, 2] = cam_infos["intrinsic"][0, 2] * scale_factor
        cam_infos["intrinsic"][1, 1] = cam_infos["intrinsic"][1, 1] * scale_factor
        cam_infos["intrinsic"][1, 2] = cam_infos["intrinsic"][1, 2] * scale_factor

        outputs["intrinsic"] = torch.from_numpy(cam_infos["intrinsic"])  # (3,3)

        outputs["images_input"] = images_input  # (V, 3, H, W)
        outputs["cam_poses_input"] = cam_poses_input  # (V, 4)
        outputs["cam_poses_4x4_input"] = cam_poses_4x4_input  # (V, 4, 4)
        outputs["images_output"] = (
            images
            if not self.opt.exclude_input_views
            else images[self.opt.num_input_views :]
        )

        outputs["masks_output"] = (
            masks
            if not self.opt.exclude_input_views
            else masks[self.opt.num_input_views :]
        )
        outputs["cam_poses_output"] = (
            cam_poses
            if not self.opt.exclude_input_views
            else cam_poses[self.opt.num_input_views :]
        )  # (V, 4)
        outputs["cam_poses_4x4_output"] = (
            cam_poses_4x4
            if not self.opt.exclude_input_views
            else cam_poses_4x4[self.opt.num_input_views :]
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

    def _load_image(self, image_name: str) -> Tensor:
        img = cv2.imread(image_name)[..., ::-1]
        img = cv2.resize(img, (self.opt.input_size, self.opt.input_size))
        image_plt = (img[..., :3] / 255).astype(np.float32)  # (H, W, 3) in [0, 1]
        return (
            torch.from_numpy(image_plt).permute(2, 0, 1).float()
        )  # (3, H, W) in [0, 1]

    def _load_mask(self, mask_name: str) -> Tensor:
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.opt.input_size, self.opt.input_size))
        mask = (mask / 255).astype(np.float32)  # (H, W, 1) in [0, 1]
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W) in [0, 1]
        return mask

    def _load_camera_from_json(self, json_bytes: bytes) -> Tensor:
        json_dict = json.loads(json_bytes)
        c2w = np.eye(4)
        c2w[:3, 0] = np.array(json_dict["x"])
        c2w[:3, 1] = np.array(json_dict["y"])
        c2w[:3, 2] = np.array(json_dict["z"])
        c2w[:3, 3] = np.array(json_dict["origin"])
        return torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)

    def _load_camera_from_pkl(self, pkl_file_path):
        with open(pkl_file_path, "rb") as file:
            data_pkl = pickle.load(file)
        new_data_npz = {}
        new_data_npz["R"] = data_pkl["R"]
        new_data_npz["T"] = data_pkl["T"]
        focal_length = data_pkl["focal_length"]
        principal_point = data_pkl["principal_point"]
        fx = fy = focal_length * data_pkl["image-size"] / 2
        cx = principal_point[0][0] + data_pkl["image-size"] / 2
        cy = principal_point[0][1] + data_pkl["image-size"] / 2
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        new_data_npz["intrinsic"] = intrinsic_matrix
        return new_data_npz

    def _get_pose(self, c2w: Tensor) -> Tensor:
        R, T = c2w[:3, :3], c2w[:3, 3]
        at = torch.tensor([0.0, 0.0, 0.0], device=R.device)
        eye = -R.transpose(0, 1) @ T
        dist = eye.norm()
        elev = torch.pi / 2 - torch.asin(eye[1] / dist)
        azim = torch.atan2(eye[2], eye[0])
        target_T = torch.tensor(
            [
                elev.item(),
                azim.item(),
                (np.log(dist.item()) - np.log(1.4))
                / (np.log(2.0) - np.log(1.4))
                * torch.pi,
                torch.tensor(0),
            ]
        )
        return target_T


if __name__ == "__main__":
    from src.options import opt_dict
    device = "cuda"
    opt = opt_dict["humansplat"]
    opt.input_size = 512
    opt.output_size = 512
    val_dataset = HumanDataset(opt, training=False, data_root=opt.thuman2_dir)
    ret = val_dataset[0]

    print(ret.keys())

    # projector = Projector(device)
    # rgb_sampled = projector.compute(
    #     ret["smpl_xyz"].unsqueeze(0),
    #     ret["intrinsic"],
    #     opt.input_size,
    #     opt.input_size,
    #     ret["images_output"],
    #     ret["origin_cam_pose"],
    #     ret["smpl_semantic"],
    #     verbose=False,
    # )

    # import pdb; pdb.set_trace()
    # xyz_color = rgb_sampled[0].permute(2,0,1).mean(-1)
    # mesh = trimesh.Trimesh(vertices=ret["smpl_xyz"])
    # import pdb; pdb.set_trace()
    # mesh.visual.vertex_colors = ret["smpl_semantic"][:, ::-1].astype(np.float32) / 255.0
    # mesh.visual.vertex_colors = xyz_color.numpy()*255
    # mesh.export('output_mesh.obj')
