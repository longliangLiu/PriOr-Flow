import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicUpdateBlock, BasicMultiUpdateBlock
from core.extractor import BasicEncoder
from core.corr import DCCL
from core.utils.utils import bilinear_sampler, coords_grid, upflow8, downflow8, cycle_bilinear_sampler
from core.utils import projection_prim_ortho


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class PriOr_RAFT(nn.Module):
    def __init__(self, args):
        super(PriOr_RAFT, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)

        self.ODDC = BasicMultiUpdateBlock(self.args, hidden_dim=hdim)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()
                
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

    def groupwise_corr(self, fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return cost

    def load_things_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = self.state_dict()
        ckpt = {}
        for key, value in checkpoints.items():
            if key.startswith('module.'):
                key = key[7:]
                ckpt[key] = value
        
        for key in state_dict.keys():
            if (key in ckpt.keys()) and (state_dict[key].shape == ckpt[key].shape):
                state_dict[key] = ckpt[key]
            elif 'ODDC' in key:
                if (('.gru.' in key) or ('.flow_head.' in key) or ('.mask.' in key)) and (state_dict[key].shape == ckpt[key.replace('ODDC', 'update_block')].shape):
                    state_dict[key] = ckpt[key.replace('ODDC', 'update_block')]
                else:
                    print(f"Skip loading parameter: {key}, not found in checkpoint")
            else:
                print(f"Skip loading parameter: {key}, not found in checkpoint")
        self.load_state_dict(state_dict, strict=True)        


    def forward(self, image1, image2, iters=12, init_flow=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1_A = image1.contiguous()
        image2_A = image2.contiguous()
        B, _, H, W = image1_A.shape

        rotate_matrix = projection_prim_ortho.generate_rotation_metrix(theta_list=[0., 0., -np.pi / 2])
        sample_grid_A2B = projection_prim_ortho.generate_samplegrid(image1_A.shape, rotate_matrix)
        sample_grid_A2B_8x = projection_prim_ortho.generate_samplegrid([B, 3, H // 8, W // 8], rotate_matrix)
        sample_grid_A2B_W2C = projection_prim_ortho.generate_samplegrid(image1_A.shape, rotate_matrix.T)
        sample_grid_A2B_W2C_8x = projection_prim_ortho.generate_samplegrid([B, 3, H // 8, W // 8], rotate_matrix.T)

        rotate_matrix = projection_prim_ortho.generate_rotation_metrix(theta_list=[0., 0., np.pi / 2])
        sample_grid_B2A = projection_prim_ortho.generate_samplegrid(image1_A.shape, rotate_matrix)
        sample_grid_B2A_8x = projection_prim_ortho.generate_samplegrid([B, 3, H // 8, W // 8], rotate_matrix)
        sample_grid_B2A_W2C = projection_prim_ortho.generate_samplegrid(image1_A.shape, rotate_matrix.T)
        sample_grid_B2A_W2C_8x = projection_prim_ortho.generate_samplegrid([B, 3, H // 8, W // 8], rotate_matrix.T)

        image1_B, image2_B = projection_prim_ortho.img_rotate(torch.cat([image1, image2], dim=1), sample_grid=sample_grid_A2B).split([3, 3], dim=1)
        image1_B = image1_B.contiguous()
        image2_B = image2_B.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet_A, cnet_B = self.cnet([image1_A, image1_B])
            net_A, inp_A = torch.split(cnet_A, [hdim, cdim], dim=1)
            net_A = torch.tanh(net_A)
            inp_A = torch.relu(inp_A)

            net_B, inp_B = torch.split(cnet_B, [hdim, cdim], dim=1)
            net_B = torch.tanh(net_B)
            inp_B = torch.relu(inp_B)

        with autocast(enabled=self.args.mixed_precision):
            fmap1_A, fmap2_A, fmap1_B, fmap2_B = self.fnet([image1_A, image2_A, image1_B, image2_B])
            fmap1_A = fmap1_A.float()
            fmap2_A = fmap2_A.float()
            fmap1_B = fmap1_B.float()
            fmap2_B = fmap2_B.float()

        cost_volume_A = self.corr(fmap1_A, fmap2_A)  # [B, H1, W1, H2, W2]
        cost_volume_B = self.corr(fmap1_B, fmap2_B)  # [B, H1, W1, H2, W2]
        
        init_cost_volume_A = cost_volume_A
        init_cost_volume_B = cost_volume_B
        
        corr_fn = DCCL(radius=self.args.corr_radius)
        corr_pyramid_A = corr_fn.build_pyramid(init_cost_volume_A)
        corr_pyramid_B = corr_fn.build_pyramid(init_cost_volume_B)
        
        coords0_A, coords1_A = self.initialize_flow(image1_A)
        coords0_B, coords1_B = self.initialize_flow(image1_B)
        if init_flow is not None:
            coords1_A = coords1_A + init_flow
            coords1_B = coords1_B + projection_prim_ortho.flo_rotate(init_flow, sample_grid_W2C=sample_grid_A2B_W2C_8x, sample_grid_C2W=sample_grid_A2B_8x)

        flow_predictions_A = []
        flow_predictions_B = []

        for itr in range(iters):
            coords1_A = coords1_A.detach()
            flow_A = coords1_A - coords0_A
            warped_fmap2_A = cycle_bilinear_sampler(fmap2_A, coords1_A.permute(0, 2, 3, 1))
            flaw_A = self.groupwise_corr(fmap1_A, warped_fmap2_A, num_groups=4)

            coords1_B = coords1_B.detach()
            flow_B = coords1_B - coords0_B

            flow_B_A = projection_prim_ortho.flo_rotate(flow_B, sample_grid_W2C=sample_grid_B2A_W2C_8x, sample_grid_C2W=sample_grid_B2A_8x)
            coords1_B_A = coords0_A + flow_B_A
            warped_fmap2_B_A = cycle_bilinear_sampler(fmap2_A, coords1_B_A.permute(0, 2, 3, 1))
            flaw_B_A = self.groupwise_corr(fmap1_A, warped_fmap2_B_A, num_groups=4)

            with autocast(enabled=self.args.mixed_precision):
                corr_A, corr_B_A = corr_fn(coords1_A, corr_pyramid_A, corr_pyramid_B, sample_grid_A2B_W2C_8x, sample_grid_B2A_8x)  # index correlation volume
                corr_B, corr_A_B = corr_fn(coords1_B, corr_pyramid_B, corr_pyramid_A, sample_grid_B2A_W2C_8x, sample_grid_A2B_8x)  # index correlation volume
                corr_A = corr_A + corr_B_A
                corr_B = corr_B + corr_A_B
                
                net_A, up_mask_A, delta_flow_A = self.ODDC(net_A, inp_A, flow_A, corr_A, flaw_A, flow_B_A, flaw_B_A)
                net_B, up_mask_B, delta_flow_B = self.update_block(net_B, inp_B, corr_B, flow_B)

            coords1_A = coords1_A + delta_flow_A
            flow_down_A = coords1_A - coords0_A

            coords1_B = coords1_B + delta_flow_B
            flow_down_B = coords1_B - coords0_B 

            # upsample predictions
            if up_mask_A is None:
                flow_up_A = upflow8(flow_down_A)
            else:
                flow_up_A = self.upsample_flow(flow_down_A, up_mask_A)

            if up_mask_B is None:
                flow_up_B = upflow8(flow_down_B)
            else:
                flow_up_B = self.upsample_flow(flow_down_B, up_mask_B)
            
            flow_predictions_A.append(flow_up_A)
            flow_predictions_B.append(flow_up_B)
        if test_mode:
            return flow_up_A
        
        return flow_predictions_A, flow_predictions_B
    
        
