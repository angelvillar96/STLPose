"""
COCO dataset

EnhancePseEstimation/src/data
Adapted from: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import os
import sys
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.append("..")

from lib.transforms import get_affine_transform
from lib.transforms import affine_transform
from lib.transforms import fliplr_joints
import CONSTANTS
from CONFIG import CONFIG


class JointsDataset(Dataset):
    """ Base class for human pose estimation datase """

    def __init__(self, exp_data, root, img_path, labels_path, is_train,
                 perceptual_loss_dict=None, transform=None):
        """ Module initializer """
        self.num_joints = 17
        self.pixel_std = 200
        self.flip_pairs = CONSTANTS.FLIP_PAIRS
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.img_path = img_path
        self.labels_path = labels_path
        self.perceptual_loss_dict = perceptual_loss_dict
        if(is_train):
            test_dir = "train2017"
        else:
            test_dir = "val2017"
        self.original_image_path = os.path.join(root, "original_images", test_dir)

        self.scale_factor = exp_data["dataset"]["scale_factor"]
        self.rotation_factor = exp_data["dataset"]["rot_factor"]
        self.flip = exp_data["dataset"]["flip"]
        self.num_joints_half_body = exp_data["dataset"]["num_joints_half_body"]
        self.prob_half_body = exp_data["dataset"]["prob_half_body"]

        self.target_type = "gaussian"
        self.image_size = np.array([192, 256])
        self.heatmap_size = np.array([48, 64])
        self.sigma = 2
        self.use_different_joints_weight = True
        self.joints_weight = 1

        self.transform = transform
        self.db = []

    def _get_db(self):
        """ """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """ """
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        """
        Half-body augmentation. If there are enought keypoints present,
        we choose either the lower or the upper body only

        Args:
        -----
        joints: list/np.array
            Human keypoints annotations
        joints_vis: list/np.array
            Visibility of each of the joints

        Returns:
        --------
        center, scale: numpy arrays
            Center coordinate and relative scale of the upper/lower body wrt original crop
        """
        upper_joints, lower_joints = [], []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        # making sure there are enough keypoints
        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints
        if len(selected_joints) < 2:
            return None, None

        # zooming into the selected joints area
        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array([
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ], dtype=np.float32)
        scale = scale * 1.5
        return center, scale

    def __len__(self):
        """ """
        return len(self.db)

    def __getitem__(self, idx):
        """ Sampling image from the dataset, keypoint annotations, and metadata """
        db_rec = copy.deepcopy(self.db[idx])  # TODO
        image_file = db_rec['image']
        image_name = os.path.basename(image_file)
        alpha = float(db_rec['alpha']) if 'alpha' in db_rec.keys() else 0.
        original_image_file = db_rec['original_image'] if 'original_image' in db_rec.keys() else ""
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        image_id = int(original_image_file[-16:-4])  # removing the png from name

        # obtaining precomputed perceptual_loss
        perceptual_loss = 0
        if self.perceptual_loss_dict:
            perceptual_loss = self.perceptual_loss_dict[image_name]

        # loading image
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        # annotations and transformation parameters
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        # applying transformations and augmentations to image and annotations
        if self.is_train:
            # applying half-body transform
            vis_joints = np.sum(joints_vis[:, 0])
            if(vis_joints > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(joints, joints_vis)
                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            # scaling and rotation augmentations
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if random.random() <= 0.6:
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
            else:
                r = 0

            # mirroring left-right the image, left-rigth annotation pairs, and center coord
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        # applying affine transformations to img/annotations as well as others (e.g. toTensor())
        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR
            )
        if self.transform:
            input = self.transform(input)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # converting kpts to heatmaps and  transforming to torch.Tensor.double()
        target, target_weight = self.generate_target(joints, joints_vis)
        input = input.double()
        target = torch.from_numpy(target).double()
        target_weight = torch.from_numpy(target_weight).double()

        meta = {
            'image': image_file,
            'original_image': original_image_file,
            'image_id': image_id,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints.astype(np.double),
            'joints_vis': joints_vis.astype(np.double),
            'center': c.astype(np.double),
            'scale': s.astype(np.double),
            'rotation': r,
            'score': score,
            'alpha': alpha,
            "perceptual_loss": perceptual_loss,
            'full_image': '',       # for compatibility in 'Combined-db'
            'image_path': '',       # for compatibility in 'Combined-db'
            'original_joints': np.zeros((17, 3)).astype(np.double),  # for compatibility
            'archdata_joints': np.zeros((18, 3)).astype(np.double),  # for compatibility
            'character_name': ''    # for compatibility in 'Combined-db'
        }
        return input, target, target_weight, meta

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

#
