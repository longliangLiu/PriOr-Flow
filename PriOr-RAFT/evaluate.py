import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import frame_utils
from tqdm import tqdm
from core.prior_raft import PriOr_RAFT
from utils.utils import InputPadder, forward_interpolate
from core.utils import polemask
from core.utils.spherical import *


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_city_regions(model, iters=24):
    """ Peform validation using the city """

    val_dataset = datasets.City_100(choice='A')

    uniform_mask = None
    regions_mask = {"all": None,
                    "nopole": None,
                    "pole": None,
                    "center": None}
    regions_results = {"all": {"epe": None, "sd": None, "sd_uni": None},
                       "nopole": {"epe": None, "sd": None, "sd_uni": None},
                       "pole": {"epe": None, "sd": None, "sd_uni": None},
                       "center": {"epe": None, "sd": None, "sd_uni": None}}
    for region in regions_mask:
        out_list, epe_list, sd_list, sd_uni_list = [], [], [], []
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            if (regions_mask[region] is None) or (uniform_mask is None):
                _, _, H, W = image1.shape
                uniform_mask = spherical_mask(H, W)
                uniform_mask = torch.from_numpy(uniform_mask).cpu()

                mask_all = torch.ones((H, W), dtype=torch.long)
                mask_pole, mask_center = polemask.generate_polemask(H, W)  # 1 x H x W
                mask_nopole = 1 - mask_pole
                regions_mask["all"] = mask_all.view(-1) >= 0.5
                regions_mask["nopole"] = mask_nopole.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["pole"] = mask_pole.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["center"] = mask_center.squeeze(0).cpu().view(-1) >= 0.5

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            sd = calculate_great_circle_distance(flow[None].cuda(), flow_gt[None].cuda())[0].cpu()
            epe_list.append(epe.view(-1)[regions_mask[region]].numpy())
            sd_list.append(sd.view(-1)[regions_mask[region]].numpy())
            sd_uni_perImg = (sd * uniform_mask).view(-1)
            sd_uni_perImg = sd_uni_perImg[regions_mask[region]] / torch.sum(uniform_mask.view(-1)[regions_mask[region]])
            sd_uni_perImg = torch.sum(sd_uni_perImg).item()
            sd_uni_list.append(sd_uni_perImg)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        sd_list = np.array(sd_list)
        sd = np.mean(sd_list)
        sd_uni_list = np.array(sd_uni_list)
        sd_uni = np.mean(sd_uni_list)

        regions_results[region] = {"epe": epe, "sd": sd, "sd_uni": sd_uni}

    for region in regions_results:
        print(f"{region: >6}-city: epe {regions_results[region]['epe']: .3f}, sd {regions_results[region]['sd']: .8f}, sd_uni {regions_results[region]['sd_uni']: .8f}")


@torch.no_grad()
def validate_MPF_regions(model, iters=12, scene='EFT'):
    """ Peform validation on different regions using the MPFDataset """

    val_dataset = datasets.MPFDataset(split='test', scene=scene)

    regions_mask = {"All": None,
                    "Equator": None,
                    "Poles": None,
                    "Center": None}
    regions_results = {"All": {"epe": None, "sd": None},
                       "Equator": {"epe": None, "sd": None},
                       "Poles": {"epe": None, "sd": None},
                       "Center": {"epe": None, "sd": None}}
    for region in regions_mask:
        epe_list, sd_list = [], []
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            if (regions_mask[region] is None):
                _, _, H, W = image1.shape

                mask_all = torch.ones((H, W), dtype=torch.long)
                mask_pole, mask_center = polemask.generate_polemask(H, W)  # 1 x H x W
                mask_equator = 1 - mask_pole
                regions_mask["All"] = mask_all.view(-1) >= 0.5
                regions_mask["Equator"] = mask_equator.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["Poles"] = mask_pole.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["Center"] = mask_center.squeeze(0).cpu().view(-1) >= 0.5

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            sd = calculate_great_circle_distance(flow[None].cuda(), flow_gt[None].cuda())[0].cpu()
            epe_list.append(epe.view(-1)[regions_mask[region]].numpy())
            sd_list.append(sd.view(-1)[regions_mask[region]].numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        sd_list = np.array(sd_list)
        sd = np.mean(sd_list)

        regions_results[region] = {"epe": epe, "sd": sd}
        print(f"{region:>7}-{scene}: epe {regions_results[region]['epe']: .3f}, sd {regions_results[region]['sd']: .8f}")

    return regions_results


@torch.no_grad()
def validate_FlowScape_regions(model, iters=12, scene='sunny'):
    """ Peform validation on different regions using the MPFDataset """

    val_dataset = datasets.FlowScape(split='test', scene=scene)

    regions_mask = {"All": None,
                    "Equator": None,
                    "Poles": None,
                    "Center": None}
    regions_results = {"All": {"epe": None, "sd": None},
                       "Equator": {"epe": None, "sd": None},
                       "Poles": {"epe": None, "sd": None},
                       "Center": {"epe": None, "sd": None}}
    for region in regions_mask:
        epe_list, sd_list = [], [], []
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            if (regions_mask[region] is None):
                _, _, H, W = image1.shape

                mask_all = torch.ones((H, W), dtype=torch.long)
                mask_pole, mask_center = polemask.generate_polemask(H, W)  # 1 x H x W
                mask_equator = 1 - mask_pole
                regions_mask["All"] = mask_all.view(-1) >= 0.5
                regions_mask["Equator"] = mask_equator.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["Poles"] = mask_pole.squeeze(0).cpu().view(-1) >= 0.5
                regions_mask["Center"] = mask_center.squeeze(0).cpu().view(-1) >= 0.5

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            sd = calculate_great_circle_distance(flow[None].cuda(), flow_gt[None].cuda())[0].cpu()
            epe_list.append(epe.view(-1)[regions_mask[region]].numpy())
            sd_list.append(sd.view(-1)[regions_mask[region]].numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        sd_list = np.array(sd_list)
        sd = np.mean(sd_list)

        regions_results[region] = {"epe": epe, "sd": sd}
        print(f"{region:>7}-FlowScape-{scene}: epe {regions_results[region]['epe']: .3f}, sd {regions_results[region]['sd']: .8f}")

    return regions_results


@torch.no_grad()
def validate_MPF(model, iters=12, scene='EFT'):
    """ Peform validation using the MPFDataset """
    model.eval()
    val_dataset = datasets.MPFDataset(split='test', scene=scene)
    epe_list, sd_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        sd = calculate_great_circle_distance(flow[None].cuda(), flow_gt[None].cuda())[0].cpu()
        epe_list.append(epe.view(-1).numpy())
        sd_list.append(sd.mean().numpy())


    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    sd_list = np.array(sd_list)
    sd = np.mean(sd_list)
    print("Validation (%s) EPE: %f, SEPE: %f" % (scene, epe, sd))

    return {f'{scene}-epe': epe, f'{scene}-SEPE': sd}


@torch.no_grad()
def validate_FlowScape(model, iters=12, scene='sunny'):
    """ Peform validation using the FlowScape """
    model.eval()
    val_dataset = datasets.FlowScape(split='test', scene=scene)
    epe_list, sd_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        sd = calculate_great_circle_distance(flow[None].cuda(), flow_gt[None].cuda())[0].cpu()
        epe_list.append(epe.view(-1).numpy())
        sd_list.append(sd.mean().numpy())


    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    sd_list = np.array(sd_list)
    sd = np.mean(sd_list)
    print("Validation (%s) EPE: %f, SEPE: %f" % (f'FlowScape-{scene}', epe, sd))

    return {f'FlowScape-{scene}-epe': epe, f'FlowScape-{scene}-SEPE': sd}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', choices=['MPFDataset', 'FlowScape'], help="dataset for evaluation")
    parser.add_argument('--scene', default='EFT', choices=['City', 'EFT', 'cloud', 'fog', 'rain', 'sunny', 'all'], help="scene for FlowScape evaluation")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--regions', action='store_true')
    args = parser.parse_args()

    model = torch.nn.DataParallel(PriOr_RAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(args.model), strict=True)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'MPFDataset':
            assert args.scene in ['City', 'EFT', 'all']
            if args.regions:
                validate_MPF_regions(model)
            else:
                validate_MPF(model, scene=args.scene)
        
        elif args.dataset == 'FlowScape':
            assert args.scene in ['cloud', 'fog', 'rain','sunny', 'all']
            if args.regions:
                validate_FlowScape_regions(model, scene=args.scene)
            else:
                validate_FlowScape(model, scene=args.scene)
