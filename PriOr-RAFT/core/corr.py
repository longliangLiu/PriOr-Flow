import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid, cycle_bilinear_sampler
from core.utils import projection_prim_ortho

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
    

class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2 ** i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class DCCL:
    def __init__(self, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

    def build_pyramid(self, cost_volume_8):
        corr_pyramid = []
        # all pairs correlation
        corr = cost_volume_8.unsqueeze(dim=3)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        # init_cost_volume = init_cost_volume
        corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_pyramid.append(corr)

        return corr_pyramid

    def __call__(self, coords, corr_pyramid_A, corr_pyramid_B, sample_grid_A2B_W2C_8x, sample_grid_B2A_8x):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid_A = []
        out_pyramid_B = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)

            coords_lvl = centroid_lvl + delta_lvl

            corr_A = cycle_bilinear_sampler(corr_pyramid_A[i], coords_lvl)
            corr_A = corr_A.view(batch, h1, w1, -1)
            out_pyramid_A.append(corr_A)

            coords_lvl = coords_lvl.reshape(batch, h1*w1, (2*r+1)**2, 2)
            coords_lvl_B = cycle_bilinear_sampler(sample_grid_A2B_W2C_8x, coords_lvl).reshape(batch, 2, h1*w1, (2*r+1)**2)

            coords_lvl_B = coords_lvl_B.permute(0, 2, 3, 1).reshape(batch*h1*w1, 2*r+1, 2*r+1, 2)
            corr_B = cycle_bilinear_sampler(corr_pyramid_B[i], coords_lvl_B)
            corr_B = corr_B.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
            corr_B = projection_prim_ortho.img_rotate(corr_B, sample_grid=sample_grid_B2A_8x)
            corr_B = corr_B.permute(0, 2, 3, 1).reshape(batch, h1, w1, -1)
            out_pyramid_B.append(corr_B)
            
        out_A = torch.cat(out_pyramid_A, dim=-1)
        out_B = torch.cat(out_pyramid_B, dim=-1)
        return out_A.permute(0, 3, 1, 2).contiguous().float(), out_B.permute(0, 3, 1, 2).contiguous().float()
