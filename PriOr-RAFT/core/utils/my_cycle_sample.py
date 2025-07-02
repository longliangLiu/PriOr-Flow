import torch
from torch.autograd import Variable



def cycle_grid_sample(input_tensor, grid, is_grid=False):
    """
    input_tensor: B x C x H x W
    grid: B x 2 x H x W
    is_grid: if is_grid is True, then input_tensor is a grid and C == 2

    a---------------------------------------------------------------c
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |--------------(x,y)------------------------------------------- |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    |                |                                              |
    b---------------------------------------------------------------d
    """
    B, C, H, W = input_tensor.shape
    _, _, H_grid, W_grid = grid.shape
    input_tensor = input_tensor.view(B, C, -1)
    grid = grid.reshape(B, 2, -1)
    grid[:, 0, :] = grid[:, 0, :] % W  # [0, W)
    grid_floor = grid.floor()  # B x 2 x H*W
    bilinear_weights = grid - grid_floor  # B x 2 x H*W
    grid_floor = grid_floor.long()
    xw = bilinear_weights[:, 0, :]  # B x H*W
    yw = bilinear_weights[:, 1, :]  # B x H*W

    wa = (1 - xw) * (1 - yw)
    wa = wa.unsqueeze(1)
    wb = (1 - xw) * yw
    wb = wb.unsqueeze(1)
    wc = xw * (1 - yw)
    wc = wc.unsqueeze(1)
    wd = xw * yw
    wd = wd.unsqueeze(1)

    x0 = grid_floor[:, 0, :]  # -1!
    x1 = x0 + 1  # 1024!
    x0 = x0 % W
    x1 = x1 % W
    y0 = grid_floor[:, 1, :]  # -1!
    y1 = y0 + 1  # 512!
    y0 = torch.clamp(y0, min=0, max=H - 1)
    y1 = torch.clamp(y1, min=0, max=H - 1)
    y0_flat = y0 * W
    y1_flat = y1 * W

    idx_a = y0_flat + x0
    idx_a = idx_a.unsqueeze(1).repeat(1, C, 1)
    idx_b = y1_flat + x0
    idx_b = idx_b.unsqueeze(1).repeat(1, C, 1)
    idx_c = y0_flat + x1
    idx_c = idx_c.unsqueeze(1).repeat(1, C, 1)
    idx_d = y1_flat + x1
    idx_d = idx_d.unsqueeze(1).repeat(1, C, 1)

    Ia = torch.gather(input_tensor, 2, idx_a)
    Ib = torch.gather(input_tensor, 2, idx_b)
    Ic = torch.gather(input_tensor, 2, idx_c)
    Id = torch.gather(input_tensor, 2, idx_d)

    if is_grid is True:
        adjust_sample_m(Ia, Ib, Ic, Id, W)

    sample_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id

    sampled = sample_flat.reshape(B, C, H_grid, W_grid)

    return sampled.contiguous()


def adjust_sample_m(Ia, Ib, Ic, Id, W):
    """
    Ia: B x 2 x H*W
    """
    Ia_m = Ia[:, 0, :]
    Ib_m = Ib[:, 0, :]
    Ic_m = Ic[:, 0, :]
    Id_m = Id[:, 0, :]

    Ib_m = Ia_m + ((Ib_m - Ia_m) + W / 2) % W - W / 2
    Ic_m = Ia_m + ((Ic_m - Ia_m) + W / 2) % W - W / 2
    Id_m = Ia_m + ((Id_m - Ia_m) + W / 2) % W - W / 2

    Ib[:, 0, :] = Ib_m
    Ic[:, 0, :] = Ic_m
    Id[:, 0, :] = Id_m


def my_cycle_warp(x, flo):
    x = x.cuda()
    flo = flo.cuda()
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid = grid.cuda()
    vgrid = grid + flo  # B,2,H,W

    x_warped = cycle_grid_sample(x, vgrid)

    return x_warped
