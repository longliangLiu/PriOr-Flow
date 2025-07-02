import argparse
import torch
from core.prior_raft import PriOr_RAFT
import os
import cv2
import numpy as np
from PIL import Image
from core.utils.flow_viz import omniflow_to_image
import torch.nn.functional as F

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].cuda()

def flow_2_colormap_np(flow_tensor):
    flow_tensor = flow_tensor.cpu().detach().squeeze()
    return omniflow_to_image(flow_tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./checkpoints/EFT/EFT-final.pth', help="restore checkpoint")
    parser.add_argument('--img1', type=str, default='./demo-frames/frame1.png', help="path of image1")
    parser.add_argument('--img2', type=str, default='./demo-frames/frame2.png', help="path of image2")

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    model = torch.nn.DataParallel(PriOr_RAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(args.model), strict=True)
    model.cuda()
    model.eval()

    image1 = load_image(args.img1)
    image2 = load_image(args.img2)

    with torch.no_grad():
        flow_pr = model(image1, image2, iters=12, test_mode=True)

    flow_pr_colored = flow_2_colormap_np(flow_pr[0])
    cv2.imwrite('./flow_pr.png', cv2.cvtColor(flow_pr_colored, cv2.COLOR_RGB2BGR))
    

