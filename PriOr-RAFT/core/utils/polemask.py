import torch
import numpy as np
from core.utils.projection_prim_ortho import *


@torch.no_grad()
def generate_polemask(H, W, delta_phi=np.pi / 2):
    """
    delta_phi: 中间区域对应的角度
    return: pole_mask_A 1 x H x W
            pole_mask_B 1 x H x W
    """
    max_phi = delta_phi / 2
    min_phi = -max_phi

    min_n = int(np.round(ERP.phi2n(max_phi, H=H)))
    max_n = int(np.round(ERP.phi2n(min_phi, H=H)))

    center_mask_A = torch.zeros((1, H, W))
    center_mask_A[:, min_n:max_n, :] = 1
    pole_mask_A = 1 - center_mask_A  # 1 x H x W
    pole_mask_B = img_A2B(pole_mask_A.unsqueeze(1).cuda()).squeeze(1)  # 1 x H x W
    pole_mask_B[pole_mask_B < 0.5] = 0
    pole_mask_B[pole_mask_B > 0] = 1

    return pole_mask_A.long().cuda(), pole_mask_B.long().cuda()


@torch.no_grad()
def generate_polemaskD(H, W, delta_phi=np.pi / 2):
    """
    delta_phi: 中间区域对应的角度
    return: pole_mask_A 1 x H x W
            pole_mask_D 1 x H x W
    """
    max_phi = delta_phi / 2
    min_phi = -max_phi

    min_n = int(np.round(ERP.phi2n(max_phi, H=H)))
    max_n = int(np.round(ERP.phi2n(min_phi, H=H)))

    center_mask_A = torch.zeros((1, H, W))
    center_mask_A[:, min_n:max_n, :] = 1
    pole_mask_A = 1 - center_mask_A  # 1 x H x W
    pole_mask_D = img_rotate(pole_mask_A.unsqueeze(1).cuda(), EulerAngles_zyx=[0., -np.pi / 2, 0.]).squeeze(1)
    pole_mask_D[pole_mask_D < 0.5] = 0
    pole_mask_D[pole_mask_D > 0] = 1

    return pole_mask_A.long().cuda(), pole_mask_D.long().cuda()


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.getcwd())

    from core import datasets

    import os.path as osp
    import cv2


    def write_imgtensor(imgtensor, filename):
        img = imgtensor[0].cpu().data.numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1].astype(np.uint8)  # RGB -> BGR
        cv2.imwrite(filename, img)


    plot_dir = osp.join(os.getcwd(), 'plots', 'City100')
    if not osp.exists(plot_dir):
        os.makedirs(plot_dir)

    val_dataset = datasets.City_100(choice='debug')
    image0, _, _, _ = val_dataset[0]
    _, H, W = image0.shape

    pole_mask_A, pole_mask_B = generate_polemask(H, W)  # 1 x H x W
    _, pole_mask_D = generate_polemaskD(H, W)
    pole_mask_A = pole_mask_A.unsqueeze(1).repeat(1, 3, 1, 1)  # 1 x 3 x H x W
    pole_mask_B = pole_mask_B.unsqueeze(1).repeat(1, 3, 1, 1)  # 1 x 3 x H x W
    pole_mask_D = pole_mask_D.unsqueeze(1).repeat(1, 3, 1, 1)  # 1 x 3 x H x W
    RGB_tensor = torch.tensor([255, 0, 0]).view(-1, 1).unsqueeze(2)[None].repeat(1, 1, H,
                                                                                 W).float().cuda()  # 1 x 3 x H x W
    pole_mask = pole_mask_A
    center_maskB = pole_mask_B
    nopole_mask = 1 - pole_mask
    center_maskD = pole_mask_D
    for idx in range(1):
        os.makedirs(osp.join(plot_dir, f'{idx}'), exist_ok=True)
        image1, _, _, _ = val_dataset[idx]
        image1 = image1[None].cuda()

        image1_masked = torch.where(pole_mask== 1, RGB_tensor, image1)
        write_imgtensor(0.5 * image1 + 0.5 * image1_masked, osp.join(plot_dir, f'{idx}', 'pole_mask.png'))

        image1_masked = torch.where(nopole_mask == 1, RGB_tensor, image1)
        write_imgtensor(0.5 * image1 + 0.5 * image1_masked, osp.join(plot_dir, f'{idx}', 'nopole_mask.png'))

        image1_masked = torch.where(center_maskB == 1, RGB_tensor, image1)
        write_imgtensor(0.5 * image1 + 0.5 * image1_masked, osp.join(plot_dir, f'{idx}', 'center_mask-B.png'))

        image1_masked = torch.where(center_maskD == 1, RGB_tensor, image1)
        write_imgtensor(0.5 * image1 + 0.5 * image1_masked, osp.join(plot_dir, f'{idx}', 'center_mask-D.png'))

