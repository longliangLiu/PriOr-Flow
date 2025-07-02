import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.getcwd())
from core.utils.my_cycle_sample import cycle_grid_sample


def generate_plane_grid(tensor_size):
    B, _, H, W = tensor_size

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    plane_grid = torch.cat((xx, yy), 1).float()
    plane_grid = plane_grid.cuda()  # B x 2 x H x W
    return plane_grid


def generate_rotation_metrix(axis_list=None, theta_list=None):
    if axis_list is None:
        axis_list = ['z', 'y', 'x']
    if theta_list is None:
        theta_list = [0., 0., 0.]

    R = torch.eye(3).cuda()
    for axis, theta in zip(axis_list, theta_list):
        cos_theta = torch.cos(torch.tensor(theta)).float()
        sin_theta = torch.sin(torch.tensor(theta)).float()
        if axis == 'x':
            R = R @ torch.tensor(
                [[1, 0, 0],
                 [0, cos_theta, -sin_theta],
                 [0, sin_theta, cos_theta]]).cuda()
        elif axis == 'y':
            R = R @ torch.tensor(
                [[cos_theta, 0, sin_theta],
                 [0, 1, 0],
                 [-sin_theta, 0, cos_theta]]).cuda()
        elif axis == 'z':
            R = R @ torch.tensor(
                [[cos_theta, -sin_theta, 0],
                 [sin_theta, cos_theta, 0],
                 [0, 0, 1]]).cuda()
    return R.cuda()


def Cartesian2Spherical(Cartesian_grid):
    """
    笛卡尔坐标系转球坐标系
    :param Cartesian_grid: torch.tensor B x 3 x H x W [x, y, z]
    :return: Spherical_grid: torch.tensor B x 2 x H x W [theta, phi]
    """
    x = Cartesian_grid[:, 0, :, :]
    y = Cartesian_grid[:, 1, :, :]
    z = Cartesian_grid[:, 2, :, :]

    phi = torch.arcsin(z)

    theta = torch.atan2(diverge_zero(y), diverge_zero(x))
    # theta = torch.atan2(y, x)
    Spherical_grid = torch.cat([theta.unsqueeze(1), phi.unsqueeze(1)], 1)
    return Spherical_grid.cuda()


def diverge_zero(x, eps=1e-6):
    near_zero = x.abs() < eps

    x = x + torch.sign(x) * near_zero * eps

    return x


def Spherical2Cartesian(Spherical_grid):
    """
    球坐标系转笛卡尔坐标系
    :param Spherical_grid: torch.tensor B x 2 x H x W [theta, phi]
    :return: Cartesian_grid: torch.tensor B x 3 x H x W [x, y, z]
    """
    theta = Spherical_grid[:, 0:1, :, :]
    phi = Spherical_grid[:, 1:2, :, :]
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    Cartesian_grid = torch.cat([x, y, z], 1)
    return Cartesian_grid.to(Spherical_grid)


def bilinear_interpolate(input_tensor, grid):
    """
    循环插值，使水平方向跨越图像边界的点可以利用另一边的像素进行双线性插值
    :param input_tensor: B x C x H x W
    :param grid: B x 2 x H x W
    :return:
    """
    B, C, H, W = input_tensor.size()
    grid[:, 0, :, :] = grid[:, 0, :, :] % W  # [0, W)
    # scale grid to [-1,1]
    ##2019 code
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2 配合grid_sample这个函数的使用

    output_tensor = F.grid_sample(input_tensor, grid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(input_tensor.size())).cuda()
    mask = F.grid_sample(mask, grid, align_corners=True)

    ##2019 author
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output_tensor * mask

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = xgrid % W

    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def cycle_interpolate(input_tensor, grid):
    """
    循环插值，使水平方向跨越图像边界的点可以利用另一边的像素进行双线性插值
    :param input_tensor: B x C x H x W
    :param grid: B x 2 x H x W
    :return:
    """
    grid = grid.clone()
    B, C, H, W = input_tensor.size()
    grid[:, 0, :, :] = grid[:, 0, :, :] % W  # [0, W)
    # scale grid to [-1,1]
    ##2019 code
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W, 1) - 1.0
    # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2 配合grid_sample这个函数的使用

    input_tensor_W0 = input_tensor[:, :, :, 0].unsqueeze(3)
    input_tensor_padding = torch.cat((input_tensor, input_tensor_W0), 3)

    output_tensor = F.grid_sample(input_tensor_padding, grid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(input_tensor_padding.size())).cuda()
    mask = F.grid_sample(mask, grid, align_corners=True)

    ##2019 author
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output_tensor * mask

def cycle_interpolate_nearest(input_tensor, grid):
    """
    循环插值，使水平方向跨越图像边界的点可以利用另一边的像素进行双线性插值
    :param input_tensor: B x C x H x W
    :param grid: B x 2 x H x W
    :return:
    """
    grid = grid.clone()
    B, C, H, W = input_tensor.size()
    grid[:, 0, :, :] = grid[:, 0, :, :] % W  # [0, W)
    # scale grid to [-1,1]
    ##2019 code
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :] / max(W, 1) - 1.0
    # 取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :] / max(H - 1, 1) - 1.0

    grid = grid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2 配合grid_sample这个函数的使用

    input_tensor_W0 = input_tensor[:, :, :, 0].unsqueeze(3)
    input_tensor_padding = torch.cat((input_tensor, input_tensor_W0), 3)

    output_tensor = F.grid_sample(input_tensor_padding, grid, mode='nearest')
    mask = torch.autograd.Variable(torch.ones(input_tensor_padding.size())).cuda()
    mask = F.grid_sample(mask, grid, mode='nearest')

    ##2019 author
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output_tensor * mask


def flow2endpoint(startpoint, flow, stack=True):
    """

    :param startpoint: B x 2 x H x W
    :param flow: B x 2 x H x W [delta_m, delta_n]
    :param stack: 将起始点坐标和终点坐标堆叠在一起作为输出
    :return:
    """
    _, _, H, W = startpoint.shape
    endpoint = startpoint + flow
    endpoint_0 = (endpoint[:, 0, :, :] + 0.5) % W - 0.5
    endpoint_1 = torch.clamp(endpoint[:, 1, :, :], min=-0.5, max=H - 0.5)

    endpoint_ = torch.cat([endpoint_0.unsqueeze(1), endpoint_1.unsqueeze(1)], 1)
    if stack is True:
        start_end_grid = torch.cat([startpoint, endpoint_], 1)  # B x 4 x H x W
        return start_end_grid
    else:
        return endpoint_


def PiecewiseFun(x, Bound):
    """
    分段线性函数 [0, Bound) -> [-0.5, Bound - 0.5)
    Args:
        x: [0, Bound) 一般来说是一个对Bound取模之后的结果
        Bound: x的上边界

    Returns:

    """
    return torch.where(x >= Bound - 0.5, x - Bound, x)


def u_clip(u):
    """
    将任意水平方向上的光流值按周期性调整到[-W/2, W/2)之间
    Args:
        u: B x H x W

    Returns:

    """
    _, _, W = u.shape
    return (u + W / 2) % W - W / 2


def rotate_cartesian(cartesian_grid, rotation_metrix):
    """

    :param cartesian_grid: 笛卡尔坐标系坐标 B x 3 x H x W dim1: x y z
    :param rotation_metrix: 旋转矩阵 3 x 3
    :return: 旋转后的笛卡尔坐标 B x 3 x H x W
    """
    B, _, H, W = cartesian_grid.shape
    # R = rotation_metrix.view(1, 1, 1, 3, 3).repeat(B, H, W, 1, 1)  # B x H x W x 3 x 3
    R = rotation_metrix.view(1, 1, 1, 3, 3)  # 1 x 1 x 1 x 3 x 3
    cartesian_grid1 = cartesian_grid.permute(0, 2, 3, 1).unsqueeze(4)  # B x H x W x 3 x1
    cartesian_rotate = torch.matmul(R, cartesian_grid1)  # B x H x W x 3 x 1
    cartesian_rotate = cartesian_rotate.squeeze(4).permute(0, 3, 1, 2)  # B x 3 x H x W

    return cartesian_rotate


class ERP:
    @staticmethod
    def n2v(n, H=512):
        """
        将ERP图像的纵向采样点索引n进行归一化
        :param H:
        :param n: 采样点索引n [0, H)
        :return: v: (0, 1)
        """
        v = (n + 0.5) / H
        return v

    @staticmethod
    def v2n(v, H=512):
        """
        v->n
        :param H:
        :param v: (0, 1)
        :return: n: 采样点索引n [0, H)
        """
        n = v * H - 0.5
        return n

    @staticmethod
    def v2phi(v):
        """
        归一化坐标v转为经线上的旋转角phi
        :param v: (0, 1)
        :return: phi: (-pi/2, pi/2)
        """
        phi = (0.5 - v) * np.pi
        return phi

    @staticmethod
    def phi2v(phi):
        """
        phi->v
        :param phi: (-pi/2, pi/2)
        :return: v: (0, 1)
        """
        v = 0.5 - phi / np.pi
        return v

    @classmethod
    def n2phi(cls, n, H=512):
        """
        n->phi
        :param H:
        :param n: 采样点索引n [0, H)
        :return: phi: 经线上的旋转角(-pi/2, pi/2)
        """
        v = cls.n2v(n, H)
        phi = cls.v2phi(v)
        return phi

    @classmethod
    def phi2n(cls, phi, H=512):
        """
        phi->n
        :param H:
        :param phi: 经线上的旋转角(-pi/2, pi/2)
        :return: n: 采样点索引n [0, H)
        """
        v = cls.phi2v(phi)
        n = cls.v2n(v, H)
        return n

    @staticmethod
    def m2u(m, W=1024):
        """
        将ERP图像的横向采样点索引m进行归一化
        :param W:
        :param m: 横向采样点索引m [0, W)
        :return: u: 横向归一化坐标 (0, 1)
        """
        u = (m + 0.5) / W
        return u

    @staticmethod
    def u2m(u, W=1024):
        """
        u->m
        :param W:
        :param u: 横向归一化坐标 (0, 1)
        :return: m: 横向采样点索引m [0, W)
        """
        m = u * W - 0.5
        return m

    @staticmethod
    def u2theta(u):
        """
        横向归一化坐标u转为纬线上的旋转角theta
        :param u: 横向归一化坐标 (0, 1)
        :return: theta: 纬线上的旋转角 [-pi, pi]
        """
        theta = (u - 0.5) * 2 * np.pi
        return theta

    @staticmethod
    def theta2u(theta):
        """
        theta->u
        :param theta: 纬线上的旋转角 [-pi, pi]
        :return: u: 横向归一化坐标 (0, 1)
        """
        u = theta / (2 * np.pi) + 0.5
        return u

    @classmethod
    def m2theta(cls, m, W):
        """
        m->theta
        :param W:
        :param m: 横向采样点索引m [0, W)
        :return: theta: 纬线上的旋转角 [-pi, pi]
        """
        u = cls.m2u(m, W)
        theta = cls.u2theta(u)
        return theta

    @classmethod
    def theta2m(cls, theta, W):
        """
        theta->m
        :param W:
        :param theta: 纬线上的旋转角 [-pi, pi]
        :return: m: 横向采样点索引m [0, W)
        """
        u = cls.theta2u(theta)
        m = cls.u2m(u, W)
        return m

    @classmethod
    def plane2spherical(cls, plane_grid):
        """
            :param plane_grid: 平面图像坐标系 B x 2 x H x W
            :return: Spherical_grid: 球坐标系 B x 2 x H x W dim1: theta phi
        """
        B, _, H, W = plane_grid.shape
        m = plane_grid[:, 0, :, :]
        n = plane_grid[:, 1, :, :]
        theta = cls.m2theta(m, W)
        phi = cls.n2phi(n, H)

        Spherical_grid = torch.cat([theta.unsqueeze(1), phi.unsqueeze(1)], 1)

        return Spherical_grid.cuda()

    @classmethod
    def spherical2plane(cls, sph_grid, tgt_size=None, is_flow=False):
        tgt_size = tgt_size if tgt_size is not None else sph_grid.shape
        B, _, H, W = tgt_size
        if is_flow is False:
            theta = sph_grid[:, -2, :, :]
            phi = sph_grid[:, -1, :, :]
            m = cls.theta2m(theta, W)
            n = cls.phi2n(phi, H)

            plane_grid = torch.cat([m.unsqueeze(1), n.unsqueeze(1)], 1)
        else:
            delta_m = sph_grid[:, -2, :, :] * W / (2 * np.pi)
            delta_n = sph_grid[:, -1, :, :] * H / (-np.pi)
            plane_grid = torch.cat([delta_m.unsqueeze(1), delta_n.unsqueeze(1)], 1)

        return plane_grid


def generate_samplegrid(tensor_size, rotate_metrix):
    B, _, H, W = tensor_size
    plane_grid = generate_plane_grid(tensor_size)
    sphrical_grid = ERP.plane2spherical(plane_grid)
    cartesian_grid = Spherical2Cartesian(sphrical_grid)

    cartesian_worldgrid = rotate_cartesian(cartesian_grid, rotate_metrix)
    sphrical_worldgrid = Cartesian2Spherical(cartesian_worldgrid)

    sample_grid = ERP.spherical2plane(sphrical_worldgrid)

    return sample_grid



def flow2camera(flow_worldgrid, rotate_metrix):
    start_worldgrid = generate_plane_grid(flow_worldgrid.shape)
    end_worldgrid = flow2endpoint(start_worldgrid, flow_worldgrid, stack=False)

    end_worldgrid_sph = ERP.plane2spherical(end_worldgrid)
    end_worldgrid_car = Spherical2Cartesian(end_worldgrid_sph)
    end_cameragrid_car = rotate_cartesian(end_worldgrid_car, rotate_metrix.T)
    end_cameragrid_sph = Cartesian2Spherical(end_cameragrid_car)

    start_worldgrid_sph = ERP.plane2spherical(start_worldgrid)
    start_worldgrid_car = Spherical2Cartesian(start_worldgrid_sph)
    start_cameragrid_car = rotate_cartesian(start_worldgrid_car, rotate_metrix.T)
    start_cameragrid_sph = Cartesian2Spherical(start_cameragrid_car)

    # {delta_theta, delta_phi}
    flow_cameragrid_sph = end_cameragrid_sph - start_cameragrid_sph
    flow_cameragrid_plane = ERP.spherical2plane(flow_cameragrid_sph, is_flow=True)
    flow_cameragrid_u = u_clip(flow_cameragrid_plane[:, 0, :, :])
    flow_caremagrid_v = flow_cameragrid_plane[:, 1, :, :]
    flow_cameragrid_uv = torch.cat([flow_cameragrid_u.unsqueeze(1), flow_caremagrid_v.unsqueeze(1)], 1)

    return flow_cameragrid_uv

def generate_samplegrid_theta(tensor_size1, tensor_size2, delta_theta):
    EulerAngles_zyx = [0., 0., delta_theta]
    rotate_matrix = generate_rotation_metrix(theta_list=EulerAngles_zyx)

    plane_grid = generate_plane_grid(tensor_size1)
    sphrical_grid = ERP.plane2spherical(plane_grid)
    cartesian_grid = Spherical2Cartesian(sphrical_grid)

    cartesian_worldgrid = rotate_cartesian(cartesian_grid, rotate_matrix)
    sphrical_worldgrid = Cartesian2Spherical(cartesian_worldgrid)

    sample_grid = ERP.spherical2plane(sphrical_worldgrid, tgt_size=tensor_size2)
    return sample_grid  # B x 2 x H x W

@torch.no_grad()
def rotating_warping(src_feat, rotate_matrix, coords):
    """
    :param src_feat: B x C x H2 x W2
    :param rotate_matrix: 3 x 3
    :param coords: B x 2 x N x H1 x W1
    :return:
    """
    B, _, H2, W2 = src_feat.shape
    B, _, N, H1, W1 = coords.shape
    coords = coords.permute(0, 2, 1, 3, 4).reshape(B*N, 2, H1, W1)
    sph_grid = ERP.plane2spherical(coords)
    car_grid = Spherical2Cartesian(sph_grid)
    car_grid_rot = rotate_cartesian(car_grid, rotate_matrix)  # B*N x 3 x H1 x W1
    sph_grid_rot = Cartesian2Spherical(car_grid_rot)
    coords_rot = ERP.spherical2plane(sph_grid_rot, tgt_size=src_feat.shape)
    coords_rot = coords_rot.reshape(B, N, 2, H1, W1).permute(0, 2, 1, 3, 4)  # B x 2 x N x H1 x W1
    warped_feat = bilinear_interpolate(src_feat, coords_rot.reshape(B, 2, N*H1, W1))  # B x C x N*H1 x W1
    warped_feat = warped_feat.reshape(B, -1, N, H1, W1)  # B x C x N x H1 x W1
    return warped_feat



def img_rotate(image, EulerAngles_zyx=None, sample_grid=None):
    if sample_grid is None:
        assert EulerAngles_zyx is not None
        rotate_matrix = generate_rotation_metrix(theta_list=EulerAngles_zyx)
        sample_grid = generate_samplegrid(image.shape, rotate_matrix)
    # image_rotate = cycle_grid_sample(image, sample_grid, is_grid=False)
    image_rotate = bilinear_sampler(image, sample_grid.permute(0, 2, 3, 1))
    return image_rotate


def img_A2B(image_A):
    image_A_B = img_rotate(image_A, EulerAngles_zyx=[0., 0., -np.pi / 2])
    return image_A_B


def img_B2A(image_B):
    image_B_A = img_rotate(image_B, EulerAngles_zyx=[0., 0., np.pi / 2])
    return image_B_A

def img_rotate_theta(image, theta):
    EulerAngles_zyx = [0., 0., theta]
    return img_rotate(image, EulerAngles_zyx)


def flo_rotate(flow, EulerAngles_zyx=None, sample_grid_W2C=None, sample_grid_C2W=None):
    rotate_matrix = None
    if sample_grid_W2C is None:
        assert EulerAngles_zyx is not None
        rotate_matrix = generate_rotation_metrix(theta_list=EulerAngles_zyx)
        sample_grid_W2C = generate_samplegrid(flow.shape, rotate_matrix.T)
    start_grid_W = generate_plane_grid(flow.shape)
    end_grid_W = flow2endpoint(start_grid_W, flow, stack=False)
    start_grid_C = sample_grid_W2C
    end_grid_C = cycle_grid_sample(sample_grid_W2C, end_grid_W, is_grid=True)
    flow_C = end_grid_C - start_grid_C
    flow_C[:, 0, :, :] = u_clip(flow_C[:, 0, :, :])
    if sample_grid_C2W is None:
        sample_grid_C2W = generate_samplegrid(flow.shape, rotate_matrix)
    flow_rotate = cycle_grid_sample(flow_C, sample_grid_C2W, is_grid=False)
    return flow_rotate

def coord_rotate(coords, EulerAngles_zyx):
    rotate_matrix = generate_rotation_metrix(theta_list=EulerAngles_zyx)
    sample_grid_W2C = generate_samplegrid(coords.shape, rotate_matrix.T)
    end_grid_W = coords
    end_grid_C = cycle_grid_sample(sample_grid_W2C, end_grid_W, is_grid=True)
    sample_grid_C2W = generate_samplegrid(coords.shape, rotate_matrix)
    coords_rotate = cycle_grid_sample(end_grid_C, sample_grid_C2W, is_grid=True)
    return coords_rotate

def coord_rotate_sample_grid(coords, sample_grid_W2C=None, sample_grid_C2W=None):
    end_grid_W = coords
    end_grid_C = cycle_grid_sample(sample_grid_W2C, end_grid_W, is_grid=True)
    coords_rotate = cycle_grid_sample(end_grid_C, sample_grid_C2W, is_grid=True)
    return coords_rotate

def flo_A2B(flow_A):
    flow_A_B = flo_rotate(flow_A, EulerAngles_zyx=[0., 0., -np.pi / 2])
    return flow_A_B


def flo_B2A(flow_B):
    flow_B_A = flo_rotate(flow_B, EulerAngles_zyx=[0., 0., np.pi / 2])
    return flow_B_A

def coord_A2B(coords_A):
    coords_A_B = coord_rotate(coords_A, EulerAngles_zyx=[0., 0., -np.pi / 2])
    return coords_A_B

def coord_B2A(coords_B):
    coords_B_A = coord_rotate(coords_B, EulerAngles_zyx=[0., 0., np.pi / 2])
    return coords_B_A

def flo_rotate_theta(flow, theta):
    EulerAngles_zyx = [0., 0., theta]
    return flo_rotate(flow, EulerAngles_zyx)

if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.getcwd())
    from core import datasets
    from core.utils.warp import cycle_warp
    from tqdm import tqdm
    import os.path as osp
    import cv2
    from core.utils import frame_utils
    from core.utils.flow_viz import omniflow_to_image, save_gif

    # dataset = datasets.EFT_100()
    # plot_dir = 'plot/EFT'
    

    # for idx in tqdm(range(5)):
    #     image1, image2, flow_gt, _ = dataset[idx]
    #     save_gif(image1, image2, flow_gt, f'{idx}_original', plot_dir)
    #     image1_A = image1[None].cuda()
    #     image2_A = image2[None].cuda()
    #     flow_gt_A = flow_gt[None].cuda()

    #     image1_A_B = img_A2B(image1_A)
    #     image2_A_B = img_A2B(image2_A)
    #     flow_gt_A_B = flo_A2B(flow_gt_A)

    #     save_gif(image1_A_B[0], image2_A_B[0], flow_gt_A_B[0], f'{idx}_A_B', plot_dir)

    img_path = '/data1/lll/datasets/Omnidirection/ODVista/test/HR/009/080.png'

    img  = frame_utils.read_gen(img_path)
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img[None].cuda()
    img = img_rotate(img, EulerAngles_zyx=[120, 0., 0.])
    img_A = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img_A = img_A[..., ::-1]
    cv2.imwrite('080_A.png', img_A)
    img_B = img_A2B(img)
    img_B = img_B[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img_B = img_B[..., ::-1]
    cv2.imwrite('080_B.png', img_B)

        
            






