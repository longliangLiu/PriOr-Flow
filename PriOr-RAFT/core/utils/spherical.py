import sys

sys.path.append('core')

import numpy as np
import torch
import math
from core.utils.projection_prim_ortho import *


def spherical_mask(H, W):
    mask = torch.arange(0, H).view(-1, 1).repeat(1, W)  # H x W
    mask = ERP.n2phi(mask, H=H)  # H x W
    mask = torch.cos(mask).cpu().numpy()
    mask = mask / np.sum(mask)

    return mask


def calculate_great_circle_distance(pre_flow, gt_flow, method='Haversine', R=1):
    """
    pre_flow: B x 2 x H x W tensor
    gt_flow: B x 2 x H x W tensor
    """
    assert method in ['Haversine', 'Cosine']
    assert (pre_flow.shape == gt_flow.shape) and (pre_flow.shape[1] == 2)

    B, _, H, W = pre_flow.shape

    startpoint = generate_plane_grid(pre_flow.shape)
    pre_endpoint_plane_grid = flow2endpoint(startpoint, pre_flow, stack=False)  # B x 2 x H x W
    pre_endpoint_spherical_grid = ERP.plane2spherical(pre_endpoint_plane_grid)  # B x 2 x H x W

    gt_endpoint_plane_grid = flow2endpoint(startpoint, gt_flow, stack=False)  # B x 2 x H x W
    gt_endpoint_spherical_grid = ERP.plane2spherical(gt_endpoint_plane_grid)  # B x 2 x H x W

    # alpha: B x H x W
    if method == 'Cosine':
        cos_alpha = torch.sin(pre_endpoint_spherical_grid[:, 1, :, :]) * torch.sin(
            gt_endpoint_spherical_grid[:, 1, :, :]) + \
                    torch.cos(pre_endpoint_spherical_grid[:, 1, :, :]) * torch.cos(
            gt_endpoint_spherical_grid[:, 1, :, :]) * \
                    torch.cos(gt_endpoint_spherical_grid[:, 0, :, :] - pre_endpoint_spherical_grid[:, 0, :, :])
        alpha = torch.arccos(cos_alpha)  # B x H x W
    elif method == 'Haversine':
        haversine_alpha = haversine(gt_endpoint_spherical_grid[:, 1, :, :] - pre_endpoint_spherical_grid[:, 1, :, :]) + \
                          torch.cos(pre_endpoint_spherical_grid[:, 1, :, :]) * torch.cos(
            gt_endpoint_spherical_grid[:, 1, :, :]) * \
                          haversine(gt_endpoint_spherical_grid[:, 0, :, :] - pre_endpoint_spherical_grid[:, 0, :, :])
        alpha = haversine_inverse(haversine_alpha)  # B x H x W
    great_circle_distance = alpha * R  # B x H x W

    return great_circle_distance


def calculate_veclen_spherical(flow, R=1):
    B, _, H, W = flow.shape

    startpoint = generate_plane_grid(flow.shape)
    endpoint_plane_grid = flow2endpoint(startpoint, flow, stack=False)  # B x 2 x H x W
    
    endpoint_spherical_grid = ERP.plane2spherical(endpoint_plane_grid)  # B x 2 x H x W
    startpoint_spherical_grid = ERP.plane2spherical(startpoint)

    haversine_alpha = haversine(endpoint_spherical_grid[:, 1, :, :] - startpoint_spherical_grid[:, 1, :, :]) + \
                      torch.cos(startpoint_spherical_grid[:, 1, :, :]) * torch.cos(
        endpoint_spherical_grid[:, 1, :, :]) * \
                      haversine(endpoint_spherical_grid[:, 0, :, :] - startpoint_spherical_grid[:, 0, :, :])
    alpha = haversine_inverse(haversine_alpha)  # B x H x W
    return R * alpha


def haversine(x):
    """
    haversine函数 y=F(x)
    """
    return torch.square(torch.sin(x / 2))


def haversine_inverse(y):
    """
    haversine函数的反函数 x=F_(y) 默认x的范围为[0,pi]
    """
    return 2 * torch.arcsin(torch.sqrt(y))
