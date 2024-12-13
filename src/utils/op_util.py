import numpy as np
import roma
import torch
import torch.nn.functional as F
from kiui.op import safe_normalize
from torch import Tensor


def normalize_cam_poses_4x4(cam_poses_4x4: Tensor, i=0, norm_radius=1.5):
    _, R, C = cam_poses_4x4.shape
    assert R == C and C == 4

    dtype = cam_poses_4x4.dtype
    cam_poses_4x4 = cam_poses_4x4.float()

    radius = torch.norm(cam_poses_4x4[i, :3, 3])
    cam_poses_4x4[:, :3, 3] *= norm_radius / radius
    transform = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, norm_radius], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=cam_poses_4x4.device,
    ) @ torch.inverse(cam_poses_4x4[i])
    return (transform.unsqueeze(0) @ cam_poses_4x4).to(dtype=dtype)  # (V, 4, 4)


def get_rays_batch(poses: Tensor, h: int, w: int, fovy: float, opengl=True):
    B = poses.shape[0]

    x, y = torch.meshgrid(
        torch.arange(w, device=poses.device),
        torch.arange(h, device=poses.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs.unsqueeze(0).expand(B, -1, -1) @ poses[:, :3, :3].transpose(
        1, 2
    )  # [B, hw, 3]
    rays_o = poses[:, :3, 3].unsqueeze(1).expand(-1, h * w, -1)  # [B, hw, 3]

    rays_o = rays_o.view(B, h, w, 3)
    rays_d = safe_normalize(rays_d).view(B, h, w, 3)

    return rays_o, rays_d


def get_rays(pose: Tensor, h: int, w: int, fovy: float, opengl=True):
    x, y = torch.meshgrid(
        torch.arange(w, device=pose.device),
        torch.arange(h, device=pose.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * (-1.0 if opengl else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if opengl else 1.0),
    )  # [hw, 3]

    rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)  # [hw, 3]

    rays_o = rays_o.view(h, w, 3)
    rays_d = safe_normalize(rays_d).view(h, w, 3)

    return rays_o, rays_d


def orbit_camera_jitter(poses: Tensor, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = (
        poses[:, :3, 1]
        * strength
        * np.pi
        * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    )
    rotvec_y = (
        poses[:, :3, 0]
        * strength
        * np.pi
        / 2
        * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    )

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T

    return new_poses


def grid_distortion(images: Tensor, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        x_steps = (
            x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)
        ).clamp(
            0, 1
        )  # perturb
        x_steps = (x_steps * W).long()  # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(
                torch.linspace(
                    grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]
                )
            )
        xs = torch.cat(xs, dim=0)  # [W]

        y_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        y_steps = (
            y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)
        ).clamp(
            0, 1
        )  # perturb
        y_steps = (y_steps * H).long()  # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(
                torch.linspace(
                    grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]
                )
            )
        ys = torch.cat(ys, dim=0)  # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        grids.append(grid)

    grids = torch.stack(grids, dim=0).to(images.device)  # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images
