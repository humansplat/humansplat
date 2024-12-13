
import torch

import numpy as np
from loguru import logger
import cv2
import trimesh

from src.utils.imutils import process_image, load_img
from extensions.pixielib.pixie import PIXIE
# from extensions.pixielib.visualizer import Visualizer
from extensions.pixielib.utils.config import cfg as pixie_cfg
RENDER_NORM = True

if RENDER_NORM:
    from src.seg3d.render import Render

class HPS():
    def __init__(self, device):
        self.device = device
        self.hps = PIXIE(config=pixie_cfg, device=self.device)
        self.hps_type = "pixie"
        
        # # smplx related
        if RENDER_NORM:
            self.render = Render(size=512, device=device)
        # self.visualizer = Visualizer(render_size=1024, config = pixie_cfg, device=device, rasterizer_type='standard')
        logger.info("Initialize HPS model successfully.")


    def forward(self, img_ori):

        # 1. process image: remove background and crop
        img_icon, img_hps, img_ori, img_mask, uncrop_param = process_image(
            img_ori, self.hps_type, 512, self.device
        )

        data_dict = {
            'image': img_icon.to(self.device).unsqueeze(0),
            'ori_image': img_ori,
            'mask': img_mask,
            'uncrop_param': uncrop_param,
        }
        

        # 2. inference
        with torch.no_grad():
            preds_dict = self.hps.forward(img_hps)

        data_dict.update(preds_dict)
        data_dict['body_pose'] = preds_dict['body_pose']
        data_dict['global_orient'] = preds_dict['global_pose']
        data_dict['betas'] = preds_dict['shape']
        data_dict['smpl_verts'] = preds_dict['vertices']
        scale, tranX, tranY = preds_dict['cam'][0, :3]
        data_dict["type"] = "smplx"
        
        data_dict['scale'] = scale
        data_dict['trans'] = torch.tensor([tranX, tranY, 0.0]).unsqueeze(0).to(self.device).float()
        

        # data_dict['smpl_faces'] = torch.Tensor(self.faces.astype(np.int64)).long().unsqueeze(0).to(
        #     self.device)

        # from rot_mat to rot_6d for better optimization
        # N_body = data_dict["body_pose"].shape[1]
        # data_dict["body_pose"] = data_dict["body_pose"][:, :, :, :2].reshape(1, N_body, -1)
        # data_dict["global_orient"] = data_dict["global_orient"][:, :, :, :2].reshape(1, 1, -1)

        return data_dict


    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_rgb_image()

    def visualize_alignment(self, data):        
        smpl_verts, smpl_landmarks, smpl_joints = self.hps.smplx.forward(
                                            shape_params=data['betas'],
                                            expression_params=data['exp'],
                                            body_pose=data['body_pose'],
                                            global_pose=data['global_orient'],
                                            jaw_pose=data['jaw_pose'],
                                            left_hand_pose=data['left_hand_pose'],
                                            right_hand_pose=data['right_hand_pose'])
        smpl_verts = ((smpl_verts + data['trans']) * data['scale']).detach().cpu().numpy()[0]
        smpl_verts *= np.array([1.0, -1.0, -1.0])
        faces = self.hps.smplx.faces_tensor.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=smpl_verts, faces=faces, process=False)
        mesh.export("smpl_mesh.obj")


        if RENDER_NORM:
            image_P = data['image']
            image_F, image_B = self.render_normal(smpl_verts, faces)
            image_F = (0.5 * (1.0 + image_F[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
            image_B = (0.5 * (1.0 + image_B[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
            image_P = (0.5 * (1.0 + image_P[0].permute(1, 2, 0).detach().cpu().numpy()) * 255.0)
            
            # Horizontally concatenate the images
            image_P = cv2.cvtColor(image_P, cv2.COLOR_RGB2BGR)
            concatenated_image = np.hstack((image_P, image_F, image_B))
            cv2.imwrite("demo1_hps.png", concatenated_image )


if __name__ == '__main__':
    hps_handle = HPS("cuda:0")

    img_ori = load_img(
        "data/assets/imgs/demo1.png",
    )
    # inference    
    data_dict = hps_handle.forward(img_ori=img_ori)

    # visualize
    hps_handle.visualize_alignment(data=data_dict)