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

