import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from einops import rearrange, repeat


# convert intrinsic matric to 4x4
def intrinsic_3x3_to_4x4(K):
    assert K.shape == (3, 3), "Input matrix must be 3x3."
    K_4x4 = torch.zeros((4, 4), dtype=K.dtype, device=K.device)
    K_4x4[:3, :3] = K
    K_4x4[3, 3] = 1
    return K_4x4


class Projector:
    def __init__(self, device):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)[
            None, None, :
        ]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.0
        return normalized_pixel_locations

    # TODO: support Batch > 1
    # def compute_projections(self, xyz, train_intrinsics, train_poses):
    #     """
    #     project 3D points into cameras
    #     :param xyz: [..., 3]
    #     : train_intrinsics [..., 4, 4]
    #     : train_poses [..., 4, 4]
    #     :return: pixel locations [..., 2], mask [...]
    #     """
    #     train_intrinsics = train_intrinsics.to(xyz.device)
    #     train_poses = train_poses.to(xyz.device)
    #     train_intrinsics = train_intrinsics.float()
    #     train_poses = train_poses.float()
    #     original_shape = xyz.shape[:2]
    #     xyz = xyz.reshape(-1, 3)
    #     num_views = len(train_intrinsics)
    #     xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
    #     projections = train_intrinsics.bmm(torch.inverse(train_poses)).bmm(
    #         xyz_h.t()[None, ...].repeat(num_views, 1, 1)
    #     )  # [n_views, 4, n_points]
    #     projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
    #     pixel_locations = projections[..., :2] / torch.clamp(
    #         projections[..., 2:3], min=1e-8
    #     )  # [n_views, n_points, 2]
    #     pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
    #     mask = projections[..., 2] > 0  # a point is invalid if behind the camera
    #     pixel_locations = pixel_locations.reshape((num_views,) + original_shape + (2,))
    #     mask = mask.reshape((num_views,) + original_shape)
    #     return pixel_locations, mask

    def compute_projections(self, xyz, train_intrinsics, train_poses):
        """
        Project 3D points into 2D using camera intrinsics and poses for batches.

        Parameters:
        - xyz: Tensor of shape (B, N, 3) where B is the batch size and N is the number of 3D points.
        - train_intrinsics: Tensor of shape (B, 3, 3) representing camera intrinsic matrices.
        - train_poses: Tensor of shape (B, 4, 4) representing camera extrinsic matrices (poses).

        Returns:
        - projected_points: Tensor of shape (B, N, 2) representing 2D coordinates on the image plane for each batch.
        """
        B, N, _ = xyz.shape

        # Add a column of ones to xyz to convert to homogeneous coordinates
        ones = torch.ones((B, N, 1), device=xyz.device)
        xyz_homogeneous = torch.cat([xyz, ones], dim=2)  # Shape (B, N, 4)

        # Transform points to camera coordinates
        # Using batch matrix multiplication (bmm)
        camera_coords = torch.bmm(
            xyz_homogeneous, train_poses.transpose(1, 2)
        )  # Shape (B, N, 4)

        # Project points using the intrinsic matrix
        # Only use the first 3 columns of train_poses for the projection calculation
        camera_coords = camera_coords[:, :, :3]  # Shape (B, N, 3)
        projected_homogeneous = torch.bmm(
            camera_coords, train_intrinsics.transpose(1, 2)
        )  # Shape (B, N, 3)
        # Convert from homogeneous to Cartesian coordinates
        # Avoid division by zero by adding a small epsilon
        eps = 1e-8
        projected_points = projected_homogeneous[:, :, :2] / (
            projected_homogeneous[:, :, 2:3] + eps
        )
        return projected_points

    @torch.no_grad()
    def proj_fea(self, data, latents, height=512, width=512):
        B = latents.shape[0]
        K = data["intrinsic"].to(self.device).to(latents.dtype)
        poses = data["origin_cam_pose"][:, 0, :, :].to(latents.device).to(latents.dtype)
        xyz = data["smpl_xyz"].to(latents.device).to(latents.dtype)  # (B, 6890, 3)
        # project xyz to latent space and sample features
        # K (RMi + t)
        pixel_locations = self.compute_projections(xyz, K, poses)  # (B, 6890, 2)
        normalized_pixel_locations = self.normalize(pixel_locations, height, width)
        sampled_fea = F.grid_sample(
            latents[:, 0, :, :],
            normalized_pixel_locations.unsqueeze(1),
            align_corners=True,
        )
        sampled_fea = rearrange(sampled_fea, "b c k p -> (b k) p c")
        smpl_prior_fea = torch.cat([xyz, sampled_fea], dim=-1)  # (B,  6890 , 3+C)
        return pixel_locations, smpl_prior_fea

    @torch.no_grad()
    def compute(
        self, xyz, K_4x4, height, width, train_imgs, poses, smpl_semantic, verbose=False
    ):
        K_4x4 = intrinsic_3x3_to_4x4(K_4x4).float()
        K_4x4 = repeat(K_4x4, "n f ->  k n f", k=len(poses))
        pixel_locations, mask_in_front = self.compute_projections(
            xyz.to(self.device), K_4x4.to(self.device), poses.to(self.device)
        )
        if verbose:
            for idx, pixel_location in enumerate(pixel_locations):
                image = torch.zeros((height, width, 3), dtype=torch.uint8)
                import pdb

                pdb.set_trace()
                for spml_idx, (x, y) in enumerate(pixel_location.squeeze(0)):
                    if 0 <= x < width and 0 <= y < height:

                        if spml_idx in smpl_semantic["head"]:
                            image[int(y), int(x), :] = torch.Tensor([1, 1, 0])
                        elif (
                            spml_idx
                            in smpl_semantic["leftFoot"] + smpl_semantic["rightFoot"]
                        ):
                            image[int(y), int(x), :] = torch.Tensor([0, 0, 1])
                        elif (
                            spml_idx
                            in smpl_semantic["leftLeg"] + smpl_semantic["rightLeg"]
                        ):
                            image[int(y), int(x), :] = torch.Tensor([1, 0, 1])
                        else:
                            image[int(y), int(x), :] = torch.Tensor([1, 1, 1])

                for spml_idx, (x, y) in enumerate(pixel_location.squeeze(0)):
                    if 0 <= x < width and 0 <= y < height:
                        if (
                            spml_idx
                            in smpl_semantic["leftHandIndex1"]
                            + smpl_semantic["rightHandIndex1"]
                            + smpl_semantic["rightHand"]
                            + smpl_semantic["leftHand"]
                            + smpl_semantic["leftArm"]
                            + smpl_semantic["rightArm"]
                            + smpl_semantic["leftForeArm"]
                            + smpl_semantic["rightForeArm"]
                        ):
                            image[int(y), int(x), :] = torch.Tensor([0, 1, 1])

                vutils.save_image(
                    image.permute(2, 0, 1) + train_imgs[idx], f"./out/{idx}_img.png"
                )
                # cv2.imwrite(f'./out/{idx}.png', image.numpy())
        # return rgbs_sampled
        return pixel_locations
