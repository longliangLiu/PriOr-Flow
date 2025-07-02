# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import logging

if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

import math
import random
from glob import glob
import os.path as osp
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, FlowAugmentor_360, SparseFlowAugmentor_360


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class FlowDataset_360(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, root=None):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor_360(**aug_params)
            else:
                self.augmentor = FlowAugmentor_360(**aug_params)
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.root = root

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        _, W, _ = flow.shape
        flow[:, :, 0] = (flow[:, :, 0] + W / 2) % W - W / 2
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


# 2022_ECCV_SLOF
class Flow360(FlowDataset_360):
    def __init__(self, aug_params=None, split='train', root='/data/lll/dataset/Flow_dataset/FLOW360_train_test'):
        super(Flow360, self).__init__(aug_params, root=root)
        assert split in ['train', 'test']
        assert osp.isdir(root)
        root = osp.join(root, split)

        dirs = sorted(glob(osp.join(root, '*')))
        for dir in dirs:
            images = sorted(glob(osp.join(dir, 'frames/*.png')))[:-1]
            # for direction in ['fflows', 'bflows']:
            for direction in ['fflows']:
                flows = sorted(glob(osp.join(dir, '{}/*.npy'.format(direction))))
                for i in range(len(flows) - 1):
                    if direction == 'fflows':
                        self.image_list += [[images[i], images[i+1]]]
                        self.flow_list += [flows[i]]
                    else:
                        self.image_list += [[images[i+1], images[i]]]
                        self.flow_list += [flows[i+1]]
        assert len(self.image_list) == len(self.flow_list), f'len(image_list)!= len(flow_list): {len(self.image_list)}!= {len(self.flow_list)}'
        logging.info('Generate Flow360 dataset from {}'.format(root))

# 2023_T-ITS_PanoFlow
class FlowScape(FlowDataset_360):
    def __init__(self, aug_params=None, split='train', root='/data/lll/dataset/Flow_dataset/FlowScape', scene='all'):
        super(FlowScape, self).__init__(aug_params, root=root)
        assert split in ['train', 'test']
        assert scene in ['cloud', 'fog', 'rain', 'sunny', 'all'], f'Invalid scene: {scene}'
        assert osp.isdir(root)
        root = osp.join(root, 'Flow360', split)
        if scene == 'all':
            self._add_scene(root, 'cloud')
            self._add_scene(root, 'fog')            
            self._add_scene(root, 'rain')
            self._add_scene(root,'sunny')
        else:
            self._add_scene(root, scene)
        assert len(self.image_list) == len(self.flow_list), f'len(image_list)!= len(flow_list): {len(self.image_list)}!= {len(self.flow_list)}'
        logging.info('Generate FlowScape dataset from {}'.format(root))

    def _add_scene(self, root, scene):
        dirs_name = sorted(os.listdir(osp.join(root, scene, 'img')))
        for dir in dirs_name:
            images = sorted(glob(osp.join(root, scene, f'img/{dir}/*.jpg')))
            flows = sorted(glob(osp.join(root, scene, f'flow/{dir}/*.flo')))
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i+1]]]
                self.flow_list += [flows[i]]

# 2022_ECCV_MPFDataset
class MPFDataset(FlowDataset_360):
    def __init__(self, aug_params=None, split='train', root='/data/lll/dataset/Flow_dataset/ECCV2022MPF-net_dataset', scene='all'):
        super(MPFDataset, self).__init__(aug_params, root=root)
        assert split in ['train', 'val', 'test']
        assert osp.isdir(root)
        assert scene in ['EFT', 'City', 'all']            
        if scene == 'EFT':
            self._add_eft(root, split)
        elif scene == 'City':
            self._add_city(root, split)
        elif scene == 'all':
            self._add_eft(root, split)
            self._add_city(root, split)
        logging.info('Generate MPFDataset dataset from {} {}'.format(root, scene))

    def _add_city(self, root, split):
        if split == 'train':
            root = osp.join(root, 'City_2000_r')
        elif split == 'val':
            root = osp.join(root, 'City_200_r')
        elif split == 'test':
            root = osp.join(root, 'City_100_r')
        flow_root = osp.join(root, 'flow')
        image_root = osp.join(root, 'image')
        flows = sorted(glob(osp.join(flow_root, '*.flo')))
        images = sorted(glob(osp.join(image_root, '*.png')))
        for i in range(len(images) - 1):  # the gt of the flow is backward flow
            self.image_list += [[images[i+1], images[i]]]
            self.flow_list += [flows[i+1]]

    def _add_eft(self, root, split):
        if split == 'train':
            root = osp.join(root, 'EFTs_Car2000')
        elif split == 'val':
            root = osp.join(root, 'EFTs_Car200')
        elif split == 'test':
            root = osp.join(root, 'EFTs_Car100')
        flow_root = osp.join(root, 'flow')
        image_root = osp.join(root, 'image')
        flows = sorted(glob(osp.join(flow_root, '*.flo')))
        images = sorted(glob(osp.join(image_root, '*.png')))
        for i in range(len(images) - 1):
            self.image_list += [[images[i+1], images[i]]]
            self.flow_list += [flows[i+1]]


# 2020_ICPR_OmniFlowNet
class OmniFlowNet_Dataset(FlowDataset_360):
    def __init__(self, aug_params=None, root='/data/lll/dataset/Flow_dataset/OMNIFLOWNET_DATASET', scene='all'):
        super(OmniFlowNet_Dataset, self).__init__(aug_params, root=root)
        assert osp.isdir(root)
        assert scene in ['CartoonTree', 'Forest', 'LowPolyModels', 'all']
        if scene == 'all':
            self._add_scene(root, 'CartoonTree')
            self._add_scene(root, 'Forest')
            self._add_scene(root, 'LowPolyModels')
        else:
            self._add_scene(root, scene)
        logging.info('Generate OmniFlowNet dataset from {}'.format(root))
    def _add_scene(self, root, scene):
        dirs = sorted(glob(osp.join(root, scene, '*')))
        for dir in dirs:
            images = sorted(glob(osp.join(dir, 'images/*.png')))
            flows = sorted(glob(osp.join(dir, 'ground_truth/*.flo')))
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i+1]]]
                self.flow_list += [flows[i]]

class OmniPhotos(FlowDataset_360):
    def __init__(self, aug_params=None, root='/data/lll/dataset/Flow_dataset/OmniPhotos'):
        super(OmniPhotos, self).__init__(aug_params, root=root)
        assert osp.isdir(root)
        # only for visualization
        self.is_test = True
        scenes = sorted(glob(osp.join(root, '*')))
        for scene in scenes:
            images = sorted(glob(osp.join(scene, 'Input/*.jpg')))
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i+1]]]
                self.extra_info += [images[i]]
        logging.info('Generate OmniPhotos dataset from {}'.format(root))

class ODVista(FlowDataset_360):
    def __init__(self, aug_params=None, split='train', root='/data1/lll/datasets/Omnidirection/ODVista', resoluton='H'):
        super(ODVista, self).__init__(aug_params, root=root)
        assert split in ['train', 'test']
        assert osp.isdir(root)
        assert resoluton in ['H', 'x2', 'x4']
        self.is_test = True
        root = osp.join(root, split)
        if resoluton == 'H':
            self._add_subdir(osp.join(root, 'HR'))
        elif resoluton == 'x2':
            subdir_groups = sorted(glob(osp.join(root, 'LR_X2', '*')))
            for group in subdir_groups:
                self._add_subdir(group)
        elif resoluton == 'x4':
            subdir_groups = sorted(glob(osp.join(root, 'LR_X4', '*')))
            for group in subdir_groups:
                self._add_subdir(group)
        logging.info('Generate ODVista dataset from {}'.format(root))

    def _add_subdir(self, root):
        subdirs = sorted(glob(osp.join(root, '*')))
        for subdir in subdirs:
            images = sorted(glob(osp.join(subdir, '*.png')))
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i+1]]]
                self.extra_info += [images[i]]


class PanoVOS(FlowDataset_360):
    def __init__(self, aug_params=None, split='train', root='/data1/lll/datasets/Omnidirection/PanoVOS'):
        super(PanoVOS, self).__init__(aug_params, root=root)
        assert split in ['train', 'test', 'val']
        assert osp.isdir(root)
        self.is_test = True
        root = osp.join(root, split)

        subdirs = sorted(glob(osp.join(root, 'JPEGImages', '*')))
        for subdir in subdirs:
            images = sorted(glob(osp.join(subdir, '*.jpg')))
            for i in range(len(images) - 1):
                self.image_list += [[images[i], images[i+1]]]
                self.extra_info += [images[i]]

        logging.info('Generate PanoVOS dataset from {}'.format(root))


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/fmj/dataset/Flow_dataset/MPI-Sintel-complete/', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        if split == 'test':
            self.is_test = True
        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id
            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/data/fmj/dataset/Flow_dataset/FlyingChairs_release/data/'):
        super(FlyingChairs, self).__init__(aug_params)
        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images) // 2 == len(flows))
        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2 * i], images[2 * i + 1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/data/fmj/dataset/Flow_dataset/Flyingthings/', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows) - 1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i + 1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i + 1], images[i]]]
                            self.flow_list += [flows[i + 1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/fmj/dataset/Flow_dataset/Kitti2015'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class KITTI_12(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/data/fmj/dataset/Flow_dataset/Kitti2012/'):
        super(KITTI_12, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True
        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_0/*_11.png')))
        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]
        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='/data/fmj/dataset/Flow_dataset/hd1k_full_package/'):
        super(HD1K, self).__init__(aug_params, sparse=True)
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))
            if len(flows) == 0:
                break
            for i in range(len(flows) - 1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i + 1]]]
            seq_ix += 1


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'City':
        aug_params = {'do_flip': False}
        City = MPFDataset(aug_params, split='train', scene='City')
        train_dataset = City

    elif args.stage == 'EFT':
        aug_params = {'do_flip': False}
        EFT =  MPFDataset(aug_params, split='train', scene='EFT')
        train_dataset = EFT

    elif args.stage == 'FlowScape':
        aug_params = {'do_flip': False}
        FS = FlowScape(aug_params, split='train')
        train_dataset = FS

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

