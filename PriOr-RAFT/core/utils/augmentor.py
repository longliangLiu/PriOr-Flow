import numpy as np
import random
import math
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)
        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]
        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1
        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
        margin_y = 20
        margin_x = 50
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)
        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])
        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        return img1, img2, flow, valid


class FlowAugmentor_360:
    def __init__(self, resize_size=None, do_flip=True):
        # resize augmentation params
        if resize_size is not None:
            self.resize_size = [resize_size[1], resize_size[0]]  # [w, h] opencv format
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        # rotation augmentation params
        self.rotate_ratio = 0.2
        self.rotaton_aug_prob = 0.5
        self.asymmetric_rotaton_aug_prob = 0.0

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    def rotation_transform(self, img1, img2, flow):
        def u_clip(u, W):
            """
            将任意水平方向上的光流值按周期性调整到[-W/2, W/2)之间
            """
            return (u + W / 2) % W - W / 2
        _, wd = img1.shape[:2]
        max_rotate_pixels = np.round(self.rotate_ratio * wd)
        if np.random.rand() < self.rotaton_aug_prob:
            if np.random.rand() < self.asymmetric_rotaton_aug_prob:
                rotate_pixels1 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                rotate_pixels2 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                img1_flow_stack = np.concatenate([img1, flow], axis=2)  # H x W x 5
                img1_flow_rotated_stack = np.zeros_like(img1_flow_stack)
                img2_rotated = np.zeros_like(img2)
                for m in range(wd):
                    img1_flow_rotated_stack[:, (m + rotate_pixels1) % wd, :] = img1_flow_stack[:, m, :]
                    img2_rotated[:, (m + rotate_pixels2) % wd, :] = img2[:, m, :]
                img1_rotated, flow_rotated = np.split(img1_flow_rotated_stack, [3], axis=2)
                flow_rotated[:, :, 0] = u_clip(flow_rotated[:, :, 0] + rotate_pixels2 - rotate_pixels1, wd)
            else:
                rotate_pixels = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                imgs_flow_stack = np.concatenate([img1.astype(np.float32), img2.astype(np.float32), flow], axis=2)  # H x W x 8
                imgs_flow_rotated_stack = np.zeros_like(imgs_flow_stack)
                for m in range(wd):
                    imgs_flow_rotated_stack[:, (m + rotate_pixels) % wd, :] = imgs_flow_stack[:, m, :]
                img1_rotated, img2_rotated, flow_rotated = np.split(imgs_flow_rotated_stack, [3, 6], axis=2)
            return img1_rotated.astype(np.uint8), img2_rotated.astype(np.uint8), flow_rotated
        else:
            return img1, img2, flow

    def flip_transform(self, img1, img2, flow):
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
        return img1, img2, flow

    def resize_transform(self, img1, img2, flow):
        img1 = cv2.resize(img1, self.resize_size, interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, self.resize_size, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, self.resize_size, interpolation=cv2.INTER_LINEAR)
        scale_x = self.resize_size[0] / img1.shape[1]
        scale_y = self.resize_size[1] / img1.shape[0]
        flow = flow * [scale_x, scale_y]
        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        # img1, img2, flow = self.resize_transform(img1, img2, flow)
        img1, img2, flow = self.rotation_transform(img1, img2, flow)
        # img1, img2, flow = self.flip_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        return img1, img2, flow


class SparseFlowAugmentor_360:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)
        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]
        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1
        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
        margin_y = 20
        margin_x = 50
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)
        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])
        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)
        return img1, img2, flow, valid


class FlowAugmentor_360_ortho:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        # rotation augmentation params
        self.rotate_ratio = 0.2
        self.rotaton_aug_prob = 0.5
        self.asymmetric_rotaton_aug_prob = 0.0

    def color_transform(self, img1_A, img2_A, img1_B, img2_B):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1_A = np.array(self.photo_aug(Image.fromarray(img1_A)), dtype=np.uint8)
            img2_A = np.array(self.photo_aug(Image.fromarray(img2_A)), dtype=np.uint8)
            img1_B = np.array(self.photo_aug(Image.fromarray(img1_B)), dtype=np.uint8)
            img2_B = np.array(self.photo_aug(Image.fromarray(img2_B)), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1_A, img2_A, img1_B, img2_B], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1_A, img2_A, img1_B, img2_B = np.split(image_stack, 4, axis=0)
        return img1_A, img2_A, img1_B, img2_B

    def eraser_transform(self, img1_A, img2_A, img1_B, img2_B, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img1_A.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2_A.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2_A[y0:y0 + dy, x0:x0 + dx, :] = mean_color
            mean_color = np.mean(img2_B.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2_B[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1_A, img2_A, img1_B, img2_B

    def rotation_transform(self, img1_A, img2_A, flow_A, img1_B, img2_B, flow_B):
        def u_clip(u, W):
            """
            将任意水平方向上的光流值按周期性调整到[-W/2, W/2)之间
            """
            return (u + W / 2) % W - W / 2
        _, wd = img1_A.shape[:2]
        max_rotate_pixels = np.round(self.rotate_ratio * wd)
        if np.random.rand() < self.rotaton_aug_prob:
            if np.random.rand() < self.asymmetric_rotaton_aug_prob:
                rotate_pixels1 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                rotate_pixels2 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                img1_flow_stack_A = np.concatenate([img1_A, flow_A], axis=2)  # H x W x 5
                img1_flow_rotated_stack_A = np.zeros_like(img1_flow_stack_A)
                img2_rotated_A = np.zeros_like(img2_A)
                img1_flow_stack_B = np.concatenate([img1_B, flow_B], axis=2)  # H x W x 5
                img1_flow_rotated_stack_B = np.zeros_like(img1_flow_stack_B)
                img2_rotated_B = np.zeros_like(img2_B)
                for m in range(wd):
                    img1_flow_rotated_stack_A[:, (m + rotate_pixels1) % wd, :] = img1_flow_stack_A[:, m, :]
                    img2_rotated_A[:, (m + rotate_pixels2) % wd, :] = img2_A[:, m, :]
                    img1_flow_rotated_stack_B[:, (m + rotate_pixels1) % wd, :] = img1_flow_stack_B[:, m, :]
                    img2_rotated_B[:, (m + rotate_pixels2) % wd, :] = img2_B[:, m, :]
                img1_rotated_A, flow_rotated_A = np.split(img1_flow_rotated_stack_A, [3], axis=2)
                flow_rotated_A[:, :, 0] = u_clip(flow_rotated_A[:, :, 0] + rotate_pixels2 - rotate_pixels1, wd)
                img1_rotated_B, flow_rotated_B = np.split(img1_flow_rotated_stack_B, [3], axis=2)
                flow_rotated_B[:, :, 0] = u_clip(flow_rotated_B[:, :, 0] + rotate_pixels2 - rotate_pixels1, wd)
            else:
                rotate_pixels = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
                imgs_flow_stack = np.concatenate([img1_A.astype(np.float32), img2_A.astype(np.float32), flow_A, img1_B.astype(np.float32), img2_B.astype(np.float32), flow_B], axis=2)  # H x W x 16
                imgs_flow_rotated_stack = np.zeros_like(imgs_flow_stack)
                for m in range(wd):
                    imgs_flow_rotated_stack[:, (m + rotate_pixels) % wd, :] = imgs_flow_stack[:, m, :]
                img1_rotated_A, img2_rotated_A, flow_rotated_A, img1_rotated_B, img2_rotated_B, flow_rotated_B = np.split(imgs_flow_rotated_stack, [3, 6, 8, 11, 14], axis=2)
            return img1_rotated_A.astype(np.uint8), img2_rotated_A.astype(np.uint8), flow_rotated_A, img1_rotated_B.astype(np.uint8), img2_rotated_B.astype(np.uint8), flow_rotated_B
        else:
            return img1_A, img2_A, flow_A, img1_B, img2_B, flow_B

    def spatial_transform(self, img1_A, img2_A, flow_A):
        # randomly sample scale
        ht, wd = img1_A.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1_A = cv2.resize(img1_A, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2_A = cv2.resize(img2_A, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_A = cv2.resize(flow_A, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_A = flow_A * [scale_x, scale_y]
        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1_A = img1_A[:, ::-1]
                img2_A = img2_A[:, ::-1]
                flow_A = flow_A[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1_A = img1_A[::-1, :]
                img2_A = img2_A[::-1, :]
                flow_A = flow_A[::-1, :] * [1.0, -1.0]
        y0 = np.random.randint(0, img1_A.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1_A.shape[1] - self.crop_size[1])
        img1_A = img1_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2_A = img2_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow_A = flow_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1_A, img2_A, flow_A

    def __call__(self, img1_A, img2_A, flow_A, img1_B, img2_B, flow_B):
        # img1_A, img2_A, flow_A, img1_B, img2_B, flow_B = self.rotation_transform(img1_A, img2_A, flow_A, img1_B, img2_B, flow_B)
        img1_A, img2_A, img1_B, img2_B = self.color_transform(img1_A, img2_A, img1_B, img2_B)
        img1_A, img2_A, img1_B, img2_B = self.eraser_transform(img1_A, img2_A, img1_B, img2_B)

        # img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1_A = np.ascontiguousarray(img1_A)
        img2_A = np.ascontiguousarray(img2_A)
        flow_A = np.ascontiguousarray(flow_A)

        img1_B = np.ascontiguousarray(img1_B)
        img2_B = np.ascontiguousarray(img2_B)
        flow_B = np.ascontiguousarray(flow_B)
        return img1_A, img2_A, flow_A, img1_B, img2_B, flow_B


class SparseFlowAugmentor_360_ortho:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1_A, img2_A, img1_B, img2_B):
        image_stack = np.concatenate([img1_A, img2_A, img1_B, img2_B], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1_A, img2_A, img1_B, img2_B = np.split(image_stack, 4, axis=0)
        return img1_A, img2_A, img1_B, img2_B

    def eraser_transform(self, img1_A, img2_A, img1_B, img2_B):
        ht, wd = img1_A.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2_A.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2_A[y0:y0 + dy, x0:x0 + dx, :] = mean_color
            mean_color = np.mean(img2_B.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2_B[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1_A, img2_A, img1_B, img2_B

    def resize_sparse_flow_map(self, flow_A, valid_A, fx=1.0, fy=1.0):
        ht, wd = flow_A.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)
        flow_A = flow_A.reshape(-1, 2).astype(np.float32)
        valid_A = valid_A.reshape(-1).astype(np.float32)
        coords0 = coords[valid_A >= 1]
        flow0 = flow_A[valid_A >= 1]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]
        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]
        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)
        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1
        return flow_img, valid_img

    def spatial_transform(self, img1_A, img2_A, flow_A, valid_A):
        # randomly sample scale
        ht, wd = img1_A.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)
        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1_A = cv2.resize(img1_A, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2_A = cv2.resize(img2_A, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_A, valid_A = self.resize_sparse_flow_map(flow_A, valid_A, fx=scale_x, fy=scale_y)
        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1_A = img1_A[:, ::-1]
                img2_A = img2_A[:, ::-1]
                flow_A = flow_A[:, ::-1] * [-1.0, 1.0]
                valid_A = valid_A[:, ::-1]
        margin_y = 20
        margin_x = 50
        y0 = np.random.randint(0, img1_A.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1_A.shape[1] - self.crop_size[1] + margin_x)
        y0 = np.clip(y0, 0, img1_A.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1_A.shape[1] - self.crop_size[1])
        img1_A = img1_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2_A = img2_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow_A = flow_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid_A = valid_A[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1_A, img2_A, flow_A, valid_A

    def __call__(self, img1_A, img2_A, flow_A, valid_A, img1_B, img2_B, flow_B, valid_B):
        img1_A, img2_A, img1_B, img2_B = self.color_transform(img1_A, img2_A, img1_B, img2_B)
        img1_A, img2_A, img1_B, img2_B = self.eraser_transform(img1_A, img2_A, img1_B, img2_B)
        # img1_A, img2_A, flow_A, valid_A = self.spatial_transform(img1_A, img2_A, flow_A, valid_A)
        img1_A = np.ascontiguousarray(img1_A)
        img2_A = np.ascontiguousarray(img2_A)
        flow_A = np.ascontiguousarray(flow_A)
        valid_A = np.ascontiguousarray(valid_A)
        return img1_A, img2_A, flow_A, valid_A, img1_B, img2_B, flow_B, valid_B


# class rotation_transform:
#     def __init__(self, asymmetric_rotaton_aug_prob=0.2, rotate_ratio=0.5):
#         self.asymmetric_rotaton_aug_prob = asymmetric_rotaton_aug_prob  # 非对称旋转变换概率
#         self.rotate_ratio = rotate_ratio
#         print('rotation aug: asymmetric_rotaton_aug_prob = %f, rotate_ratio = %f' % (self.asymmetric_rotaton_aug_prob, self.rotate_ratio))
#
#     def rotation_transform(self, img1, img2, flow, theta=[int(0)]):
#         _, wd = img1.shape[:2]
#         if len(theta) == 1:  # 进行对称旋转变换
#             imgs_flow_stack = np.concatenate([img1.astype(np.float32), img2.astype(np.float32), flow], axis=2)  # H x W x 8
#             imgs_flow_rotated_stack = np.zeros_like(imgs_flow_stack)
#             # rotate_pixels = np.random.randint(-wd / 2, wd / 2)
#             if isinstance(theta[0], int):
#                 rotate_pixels = theta[0]
#                 for m in range(wd):
#                     imgs_flow_rotated_stack[:, (m + rotate_pixels) % wd, :] = imgs_flow_stack[:, m, :]
#                 img1_rotated, img2_rotated, flow_rotated = np.split(imgs_flow_rotated_stack, [3, 6], axis=2)
#         elif len(theta) == 2:  # 进行非对称旋转变换
#             img1_flow_stack = np.concatenate([img1, flow], axis=2)  # H x W x 5
#             img1_flow_rotated_stack = np.zeros_like(img1_flow_stack)
#             img2_rotated = np.zeros_like(img2)
#             if isinstance(theta[0], int):
#                 rotate_pixels1 = theta[0]
#                 rotate_pixels2 = theta[1]
#                 for m in range(wd):
#                     img1_flow_rotated_stack[:, (m + rotate_pixels1) % wd, :] = img1_flow_stack[:, m, :]
#                     img2_rotated[:, (m + rotate_pixels2) % wd, :] = img2[:, m, :]
#                 img1_rotated, flow_rotated = np.split(img1_flow_rotated_stack, [3], axis=2)
#                 flow_rotated[:, :, 0] = u_clip(flow_rotated[:, :, 0] + rotate_pixels2 - rotate_pixels1, wd)
#
#         return img1_rotated.astype(np.uint8), img2_rotated.astype(np.uint8), flow_rotated
#
#     def __call__(self, img1, img2, flow):
#         _, wd = img1.shape[:2]
#         max_rotate_pixels = np.round(self.rotate_ratio * wd)
#         # asymmetric
#         if np.random.rand() < self.asymmetric_rotaton_aug_prob:
#             rotate_pixels1 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
#             rotate_pixels2 = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
#             return self.rotation_transform(img1, img2, flow, [rotate_pixels1, rotate_pixels2])
#         # symmetric
#         else:
#             rotate_pixels = np.random.randint(-max_rotate_pixels, max_rotate_pixels)
#             return self.rotation_transform(img1, img2, flow, [rotate_pixels])


