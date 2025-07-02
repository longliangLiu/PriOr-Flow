# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np
from .projection_prim_ortho import ERP
from .spherical import calculate_veclen_spherical
import torch
import cv2
from .my_cycle_sample import my_cycle_warp
from PIL import Image
import os
import logging

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1  # 对颜色线性插值
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    # 对光流向量进行归一化 得到长度小于1的向量
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def omniflow_to_image(flow_tensor, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_tensor.ndim == 3, 'input flow must have three dimensions'
    assert flow_tensor.shape[0] == 2, 'input flow must have shape [2,H,W]'
    if clip_flow is not None:
        flow_tensor = torch.clamp(flow_tensor, 0, clip_flow)
    sd = calculate_veclen_spherical(flow_tensor[None].cuda())[0].cpu().numpy()  # H x W
    sorted_sd = np.sort(sd, axis=None)
    persentile_95_index = int(0.95 * len(sorted_sd))
    clip_sd = sorted_sd[persentile_95_index]
    sd = np.clip(sd, 0, clip_sd)

    flow_np = flow_tensor.cpu().numpy()
    u = flow_np[0,:,:]
    v = flow_np[1,:,:]
    a = np.arctan2(-v, -u) / np.pi
    
    rad = sd
    rad_max = np.max(rad)

    epsilon = 1e-5
    # 对光流向量进行归一化 得到长度小于1的向量
    rad = rad / (rad_max + epsilon)
    return omniflow_uv_to_colors(rad, a, convert_to_bgr)


def omniflow_uv_to_colors(rad, a, convert_to_bgr):
    """
        Applies the flow color wheel to (possibly clipped) flow components u and v.

        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun

        Args:
            u (np.ndarray): Input horizontal flow of shape [H,W]
            v (np.ndarray): Input vertical flow of shape [H,W]
            convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

        Returns:
            np.ndarray: Flow visualization image of shape [H,W,3]
        """
    flow_image = np.zeros((rad.shape[0], rad.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


# 2023_T-ITS_PanoFlow
def better_flow_to_image(flow_uv,
                         alpha=0.5,
                         max_flow=724,
                         clip_flow=None,
                         convert_to_bgr=False):
    """Used for visualize extremely large-distance flow"""
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = max_flow
    param_with_alpha = np.power(rad / max_flow, alpha)
    epsilon = 1e-5
    u = param_with_alpha * u / (rad_max + epsilon)
    v = param_with_alpha * v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def save_gif(image1, image2, flow_est, filename, out_folder):
    """
        Save a GIF file of the input images and estimated flow.

        Args:
            image1 (torch.Tensor): First image of shape [3, H, W]
            image2 (torch.Tensor): Second image of shape [3, H, W]
            flow_est (torch.Tensor): Estimated flow of shape [2, H, W]
            filename (str): Filename of the GIF file
            out_folder (str): Output folder of the GIF file
    """
    # image text configure
    font                   = cv2.FONT_HERSHEY_DUPLEX
    LeftTopCornerOfText    = (10, 30)
    LeftBottomCornerOfText = (10, image1.shape[1]-10)
    RightBottomCornerOfText = (image1.shape[2]-10, image1.shape[1]-10)
    fontScale              = 1
    fontColor              = (0, 0, 0)
    thickness              = 1
    lineType               = cv2.LINE_AA

    assert image1.ndim == 3 and image2.ndim == 3, "The input images must have three dimensions."
    assert image1.shape == image2.shape, "The input images have inconsistent shapes."
    assert image1.shape[-2:] == flow_est.shape[-2:], "The input flow have inconsistent shapes."

    in_h, in_w = image1.shape[-2:]
    image1_np = image1.cpu().permute(1,2,0).numpy().astype(dtype='uint8').copy()
    image2_np = image2.cpu().permute(1,2,0).numpy().astype(dtype='uint8').copy()
    
    # flow color map
    flow_est_vis = omniflow_to_image(flow_est)
    # flow_est_vis = better_flow_to_image(flow_est.permute(1,2,0).numpy(), alpha=0.1, max_flow=25)
    flow_est_vis = cv2.putText(flow_est_vis, f'flow_est', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    max_u = flow_est[0].abs().max().item()
    max_v = flow_est[1].abs().max().item()
    flow_est_vis = cv2.putText(flow_est_vis, f'max_u={max_u:.2f}', LeftBottomCornerOfText, font, fontScale, fontColor, thickness, lineType)
    text = f'max_v={max_v:.2f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
    LeftTopCornerOfText1 = (RightBottomCornerOfText[0] - text_width, RightBottomCornerOfText[1])
    flow_est_vis = cv2.putText(flow_est_vis, f'max_v={max_v:.2f}', LeftTopCornerOfText1, font, fontScale, fontColor, thickness, lineType)

    # reconstructed estimated image1
    image1_recon_est = my_cycle_warp(image2[None], flow_est[None])
    image1_recon_est = image1_recon_est.squeeze(0).permute(1,2,0).cpu().numpy().astype(dtype='uint8').copy()
    
    image1_recon_est = cv2.putText(image1_recon_est, 'image1_recon_est', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)

    # put text
    image1_np = cv2.putText(image1_np, 'image1', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)
    image2_np = cv2.putText(image2_np, 'image2', LeftTopCornerOfText, font, fontScale, fontColor, thickness, lineType)


    # Concat
    assert image1_np.shape == image1_recon_est.shape == image2_np.shape == flow_est_vis.shape, "The input images and flow have inconsistent shapes."
    all_vis = np.concatenate((image2_np, image1_recon_est, flow_est_vis), axis=1)
    ref_vis = np.concatenate((image1_np, image1_np, flow_est_vis), axis=1)
    all_vis_pil = Image.fromarray(all_vis)
    ref_vis_pil = Image.fromarray(ref_vis)
    gif_list = [all_vis_pil, ref_vis_pil]

    # save gif
    output_path = os.path.join(out_folder, filename+'.webp')
    # output_path = os.path.join(out_folder, filename+'.gif')
    parent_path = os.path.abspath(os.path.join(output_path, os.pardir))
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    
    gif_list[0].save(output_path, save_all=True, append_images=gif_list[1:], duration=500, loop=0)