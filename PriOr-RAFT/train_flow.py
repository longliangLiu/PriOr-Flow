from __future__ import print_function, division
import sys
sys.path.append('core')
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core.prior_raft import PriOr_RAFT
from core.utils import spherical
import evaluate
import core.datasets as datasets
from tqdm import tqdm
import wandb
import logging
from core.utils.flow_viz import omniflow_to_image
from core.utils import projection_prim_ortho

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def flow_2_colormap_np(flow_tensor):
    flow_tensor = flow_tensor.cpu().detach().squeeze()
    return omniflow_to_image(flow_tensor)

class uniform_loss:
    def __init__(self, H, W):
        self.uniform_mask = spherical.spherical_mask(H, W)  # H x W np.array
        self.uniform_mask = torch.from_numpy(self.uniform_mask).cuda()
        self.uniform_mask = self.uniform_mask[None]  # 1 x H x W

    def __call__(self, flow_preds, flow_gt, valid, gamma=0.8, extro_info='', max_flow=MAX_FLOW):
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        mag = torch.sum(flow_gt ** 2, dim=1).sqrt()  # B x H x W
        valid = (valid >= 0.5) & (mag < max_flow)  # B x H x W
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = torch.sum((flow_preds[i] - flow_gt).abs(), dim=1)  # B x H x W  使用L1范数
            flow_loss += i_weight * torch.sum(valid * self.uniform_mask * i_loss)

        epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
        metrics = {
            extro_info+'epe': epe.mean().item(),
            extro_info+'1px': (epe < 1).float().mean().item(),
            extro_info+'3px': (epe < 3).float().mean().item(),
            extro_info+'5px': (epe < 5).float().mean().item()
        }
        return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.05,
                                              cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def train(args):
    model = PriOr_RAFT(args).cuda()
    model = nn.DataParallel(model)

    print("Parameter Count: %d" % count_parameters(model))
    if args.restore_ckpt is not None:
        try:
            model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device('cuda')), strict=True)
        except:  # fist load pretrained flyingthings' weights
            model.module.load_things_ckpt(args.restore_ckpt)
        logging.info("Loaded checkpoint %s done." % args.restore_ckpt)

    model.train()
    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    wandb.init(project=args.project_name, name=args.name, config=args)

    should_keep_training = True
    Uni_Loss = None
    while should_keep_training:
        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow_gt, valid = [x.cuda() for x in data_blob]
            with torch.no_grad():
                flow_gt_B = projection_prim_ortho.flo_A2B(flow_gt)
                valid_B = (flow_gt_B[:, 0, :, :].abs() < 1000) & (flow_gt_B[:, 1, :, :].abs() < 1000)
                valid_B = valid_B.float()
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            flow_predictions_A, flow_predictions_B = model(image1, image2, iters=args.iters)

            if Uni_Loss is None:
                _, _, H, W = image1.shape
                Uni_Loss = uniform_loss(H, W)

            loss_A, metrics_A = Uni_Loss(flow_predictions_A, flow_gt, valid, args.gamma, extro_info='A-')
            loss_B, metrics_B = Uni_Loss(flow_predictions_B, flow_gt_B, valid_B, args.gamma, extro_info='B-')
            loss = loss_A + loss_B
            metrics = {**metrics_A, **metrics_B}
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            wandb.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_steps)
            wandb.log(metrics, total_steps)

            ####visualize the depth_mono and disp_preds
            if total_steps % 1024 == 0:
                image1_B = projection_prim_ortho.img_A2B(image1)
                image2_B = projection_prim_ortho.img_A2B(image2)

                image1_np = image1[0].squeeze().cpu().permute(1, 2, 0).numpy()
                image1_np = image1_np.astype(np.uint8)
                image2_np = image2[0].squeeze().cpu().permute(1, 2, 0).numpy()
                image2_np = image2_np.astype(np.uint8)

                image1_np_B = image1_B[0].squeeze().cpu().permute(1, 2, 0).numpy()
                image1_np_B = image1_np_B.astype(np.uint8)
                image2_np_B = image2_B[0].squeeze().cpu().permute(1, 2, 0).numpy()
                image2_np_B = image2_np_B.astype(np.uint8)
                
                flow_pred_np_A = flow_2_colormap_np(flow_predictions_A[-1][0].squeeze())
                flow_pred_np_B = flow_2_colormap_np(flow_predictions_B[-1][0].squeeze())

                flow_gt_np = flow_2_colormap_np(flow_gt[0].squeeze())
                
                wandb.log({"image1": wandb.Image(image1_np, caption="step:{}".format(total_steps))}, total_steps)
                wandb.log({"image2": wandb.Image(image2_np, caption="step:{}".format(total_steps))}, total_steps)
                wandb.log({"image1_B": wandb.Image(image1_np_B, caption="step:{}".format(total_steps))}, total_steps)
                wandb.log({"image2_B": wandb.Image(image2_np_B, caption="step:{}".format(total_steps))}, total_steps)
                wandb.log({"flow_gt": wandb.Image(flow_gt_np, caption="step:{}".format(total_steps))}, total_steps)

                wandb.log({"flow_pred_A": wandb.Image(flow_pred_np_A, caption="step:{}".format(total_steps))}, total_steps)
                wandb.log({"flow_pred_B": wandb.Image(flow_pred_np_B, caption="step:{}".format(total_steps))}, total_steps)
                

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = os.path.join(args.save_path, '%d.pth' % (total_steps + 1))
                torch.save(model.state_dict(), PATH)
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'City':
                        results.update(evaluate.validate_MPF(model.module, scene='City'))
                    elif val_dataset == 'EFT':
                        results.update(evaluate.validate_MPF(model.module, scene='EFT'))
                    elif val_dataset == 'FlowScape':
                        results.update(evaluate.validate_FlowScape(model.module))
                wandb.log(results)
                model.train()
                model.module.freeze_bn()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    wandb.finish()
    PATH = os.path.join(args.save_path, 'final.pth')
    torch.save(model.state_dict(), PATH)
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='PriOr-Flow', help="wandb project name")
    parser.add_argument('--name', default='EFT', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')
    
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

    # architecture
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--save_path', type=str, default='./checkpoints')

    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)

    workpath = os.path.dirname(os.path.abspath(__file__))
    args.workpath = workpath

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train(args)
