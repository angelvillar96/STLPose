"""
Implementation of a dataset object to fit with the Archeological data
This dataset can be used for both the tasks of person detection and pose estimation

@author: Angel Villar-Corrales
"""

import os
import sys
import json
import copy
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

sys.path.append("..")
from lib.logger import print_
from lib.transforms import get_affine_transform
from lib.transforms import affine_transform
from lib.transforms import fliplr_joints
import CONSTANTS
from CONFIG import CONFIG


class ArchDataset(Dataset):
    """
    Implementation of a Dataset object for the Archeological Data. It is fit with the
    archeolgoical images and the annotations to perform person and keypoint detection

    Args:
    -----
    task: string
        Current task to solve with this dataset: ['person_detection', 'pose_estimation']
    """

    def __init__(self, split_set="train", task="person_detection", valid_size=0.2,
                 shuffle=True, exp_data=None, resizer=None, transform=None,
                 create_split=False, percentage=None):
        """ Initializer of the ArchDataset dataset """
        assert split_set in ["train", "validation", "test"]
        assert task in ["person_detection", "pose_estimation"],  f"Unexpected task argument: {task}."

        self.exp_data = exp_data
        self.transform = transform
        self.resizer = resizer
        self.image_data = []
        self.num_images = 0
        self.shuffle = shuffle
        self.num_instances = 0
        self.valid_size = valid_size
        self.instance_data = []
        self.task = task
        self.split_set = split_set
        self.create_split = create_split
        self.percentage = percentage

        data_path = CONFIG["paths"]["data_path"]
        if(self.task == "person_detection"):
            self.data_path = os.path.join(data_path, "class_arch_data")
            self.annotations_file = os.path.join(data_path, "annotations_arch_data", "all_data.json")
        if(self.task == "pose_estimation"):
            self.data_path = os.path.join(data_path, "class_arch_poses", "characters")
            self.annotations_file = os.path.join(data_path, "annotations_arch_data", "arch_data_keypoints.json")

        # classes that correspond to person instances. Others will be filtered out
        self.filter = ["Heracles", "persecutor", "wrestler", "abductor", "abductee",
                       "Triton", "bride", "groom", "Theseus", "Antaios", "Peleus",
                       "Atalante", "Skiron", "Eros", "Thetis", "Nereus", "maenad",
                       "satyr", "Anteros", "Procrustes", "fleeing", "Kerkyon"]
        self.arch_labels_map = {}
        self.arch_labels = []

        # parameters for preprocessing the person-instance images
        self.image_size = np.array([192, 256])
        self.heatmap_size = np.array([48, 64])
        self.aspect_ratio = 192 * 1.0 / 256
        self.pixel_std = 200
        self.num_kpts = 17
        self.sigma = 2

        # augmentation parameters for the keypoint detection
        self.scale_factor = exp_data["dataset"]["scale_factor"] if exp_data is not None else 0
        self.rotation_factor = exp_data["dataset"]["rot_factor"] if exp_data is not None else 0
        self.flip = exp_data["dataset"]["flip"] if exp_data is not None else False
        self.flip_pairs = CONSTANTS.FLIP_PAIRS

        # loading annotations and obtaining desired split
        self._load_data()
        self._get_split()
        if(self.percentage is not None):
            print_(f"Sampling only {percentage}% of ArchData training data...")
            self._get_percentage_data(percentage=percentage)
        return

    def __len__(self):
        """
        Obtaining the total number of samples in the dataset

        Returns:
        --------
        n_imgs: integer
            number of images for person detection, or number of annotated person instances
            for pose estimation.
        """
        if(self.task == "person_detection"):
            n_imgs = self.num_images
        elif(self.task == "pose_estimation"):
            n_imgs = self.num_instances
        else:
            n_imgs = 0
        return n_imgs

    def __getitem__(self, idx):
        """ Obtaining a sample from the dataset, wits pose/bbox annotations """
        if(self.task == "person_detection"):
            input, meta = self._get_detection_item(idx)
            return input, meta
        elif(self.task == "pose_estimation"):
            input, target, target_weight, meta = self._get_pose_item(idx)
            return input, target, target_weight, meta
        else:
            print(f"ERROR! Unrecognized task: '{self.task}'")
        return

    def _get_detection_item(self, idx):
        """ Sampling an element from the ArchData Detection database """
        data = copy.deepcopy(self.data[idx])
        image_name = data['image_name']
        image_path = data['image_path']
        image_id = data["image_id"]
        original_image_name = data['image_name']
        targets = data["targets"]

        if(not os.path.exists(image_path)):
            print(image_path, targets, image_name)
            raise FileNotFoundError()

        data_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)

        original_image_size = data_numpy.shape
        scale = None
        if self.resizer is not None:
            data_numpy, targets, scale = self.resizer(data_numpy, targets)
        if self.transform is not None:
            data_numpy = self.transform(data_numpy)
        else:
            data_numpy = data_numpy.transpose(2, 0, 1)

        input = data_numpy
        meta = {
            'image_name': image_name,
            'original_image_name': original_image_name,  # for pipeline compatibility
            "targets": targets,
            "image_id": image_id,  # for pipeline compatibility
            "scale": scale,
            "original_size": original_image_size[:2],
            "perceptual_loss": 0                # for compatibility in 'Combined-db'
        }
        return input, meta

    def _get_pose_item(self, idx):
        """ Sampling an element from the ArchData Pose Estimation dataset """
        db = copy.deepcopy(self.data[idx])

        # loading current image
        img_path = os.path.join(self.data_path, db['image'])
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # loading ground truth and metadata
        kpts = db['kpts']
        kpt_vis = db['kpt_vis']
        archdata_kpts = db["archdata_kpts"]
        c = db['center']
        s = db['scale']
        score = db['score']
        r = 0

        # obtaining augmentations only if in training set
        if self.split_set == "train":
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
            if self.flip and random.random() <= 0.5:  # note that we dont flip archdata_kpts
                data_numpy = data_numpy[:, ::-1, :]
                kpts, kpt_vis = fliplr_joints(
                    kpts, kpt_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        # defining image transformation: matching desired shape, maintaining aspect ratio
        # and applying data augmentation to image (extra scaling and rotation)
        transf = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            transf,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        # applying parameter transforms (e.g toTensor or Normalize)
        if self.transform:
            input = self.transform(input)
        # applying transformation to annotations so as to match input image
        original_joints = np.copy(kpts)
        for i in range(self.num_kpts):
            if kpt_vis[i, 0] > 0.0:
                kpts[i, 0:2] = affine_transform(kpts[i, 0:2], transf)

        input = input.double()
        # converting keypoints (x,y,v) into a channel (H,W) with hearmap centered at (x,y)
        target, target_weight = self.generate_target(kpts, kpt_vis)
        target = torch.from_numpy(target).double()
        target_weight = torch.from_numpy(target_weight).double()

        meta = {
            'image': db['image'],
            'image_id': db['image_id'],
            'full_image': db['original_image'],
            'image_path': img_path,
            'joints': kpts.astype(np.double),
            'joints_vis': kpt_vis.astype(np.double),
            'original_joints': original_joints.astype(np.double),
            'archdata_joints': archdata_kpts.astype(np.double),
            'center': c.astype(np.double),
            'scale': s.astype(np.double),
            'rotation': r,
            'score': score,
            'character_name': db['character_name'],
            'alpha': 0.,                        # for compatibility in 'Combined-db'
            'original_image': '',               # for compatibility in 'Combined-db'
            'filename': '',                     # for compatibility in 'Combined-db'
            'imgnum': 0,                        # for compatibility in 'Combined-db'
            'perceptual_loss': 0                # for compatibility in 'Combined-db'
        }
        return input, target, target_weight, meta

    def _get_split(self):
        """ Obtaining the desired dataset split """
        # creating the dataset split randomly
        if(self.create_split):
            # obtaining the indices of the current split. Notice that we shuffle data
            # to avoid narratives being in order (otherwise)
            all_idx = np.arange(start=0, stop=self.num_images)
            np.random.seed(seed=CONFIG["random_seed"])
            np.random.shuffle(all_idx)
            split_idx = int(np.round(self.num_images * (1-self.valid_size)))
            cur_idx = all_idx[:split_idx] if(self.split_set == "train") else all_idx[split_idx:]
            self.split_ids = cur_idx

        # using the predefined cannonical dataset split (random_seed=13)
        else:
            split_dic_path = os.path.join(CONFIG["paths"]["dict_path"], "arch_data_det_splits.json")
            if(not os.path.exists(split_dic_path)):
                print_("ERROR! Dictionary with ClassArch splits does not exist.\n "
                       "Run 'aux_create_train_valid_arch_data.py' first...")
                exit()
            with open(split_dic_path) as file:
                split_dic_path = json.load(file)
            eval_idx = split_dic_path["test"]
            if(self.split_set == "train"):
                all_idx = np.arange(start=0, stop=self.num_images)
                cur_idx = [id for id in all_idx if id not in eval_idx]
            else:
                cur_idx = eval_idx

        # updating data and annotations with the ones from the split
        self.num_images = len(cur_idx)
        self.num_instances = len(cur_idx)
        self.data = [self.data[id] for id in cur_idx]
        return

    def _get_percentage_data(self, percentage=100):
        """
        Replaces the training set with a percentage of the total data (e.g. 25 or 50)

        Args:
        -----
        percentage: float
            percentage of the images from the training set to use for training/fine-tuning
        """
        assert percentage >= 1 and percentage <= 100, "ERROR! 'Percentage' parameter "\
            f"must be in range [1, 100]. Value '{percentage}' was given."

        n_imgs = len(self.data)
        img_thr = int(np.round(n_imgs * percentage / 100))
        self.data = self.data[:img_thr]
        self.num_images = len(self.data)
        self.num_instances = len(self.data)
        return

    def _load_data(self):
        """
        Loading Data given the current taks
        """
        if(self.task == "person_detection"):
            self._load_det_data()
        elif(self.task == "pose_estimation"):
            self._load_pose_data()
        return

    def _load_det_data(self):
        """ Reading person detection annotations file to preprocess the data """
        # reading annotations file
        with open(self.annotations_file) as file:
            annotations = json.load(file)
        self.arch_labels = annotations["categories"]
        self.arch_labels_map = {lbl["id"]: lbl["name"] for lbl in self.arch_labels}
        instances = annotations["annotations"]

        # preprocessing annotations
        for inst in instances:
            xmin, ymin, xmax, ymax = [int(c) for c in inst["bbox"].split(",")]
            coords = [xmin, ymin, xmax - xmin, ymax - ymin]  # converting to x,y,w,h
            inst["bbox"] = coords

        # fitting COCO
        self.coco = COCO()
        self.coco.dataset = annotations
        self.coco.createIndex()
        self.image_set_index = self.coco.getImgIds()

        # loading and processing anntoations
        self.detection_data = self._load_coco_det_data()
        self.data = np.copy(self.detection_data)
        self.num_images = len(self.detection_data)
        return

    def _load_pose_data(self):
        """
        Loading annotations from the pose estimation annotations file to preprocess
        data and metadata
        """
        # reading annotations file
        with open(self.annotations_file) as file:
            annotations = json.load(file)

        self.coco = COCO()
        self.coco.dataset = annotations
        self.coco.createIndex()
        self.image_set_index = self.coco.getImgIds()

        # loading and processing anntoation and metadata
        self.pose_estimation_data = self._load_coco_pose_data()
        self.data = np.copy(self.pose_estimation_data)
        self.num_images = len(self.pose_estimation_data)

        return


    ##################################################################################
    # TODO
    # Some of the methods below are common to detection and pose-estimation datasets
    # maybe we should move them to a data_utils.py library to avoid code redundancies
    #################################################################################

    def _load_coco_det_data(self):
        """
        Creating a list containign the ground truth and metadata from all annotated
        instances for the task of person detection

        Returns:
        --------
        data: list
            list containing a dictionary with the targets (GT) and metadata from each
            annotated person instance
        """

        data = []
        for index in self.image_set_index:
            targets = self._process_bbox_annotations(index)
            targets["image_id"] = index
            cur_data = {}
            cur_data["image_name"] = targets["img_name"]
            cur_data["image_path"] = targets["img_path"]
            cur_data["targets"] = targets
            cur_data["image_id"] = index
            # if the image does not contain any bounding box, we skip it
            if( len(cur_data["targets"]["labels"]) == 0 or cur_data["image_name"] is None):
                continue
            # avoinding broken images
            if(not os.path.exists(cur_data["image_path"])):
                continue

            data.append(cur_data)
        return data


    def _load_coco_pose_data(self):
        """
        Creating a list containign the ground truth and metadata from all annotated
        instances for the task of pose estimation

        Returns:
        --------
        data: list
            list containing a dictionary with the targets (GT) and metadata from each
            annotated person instance
        """

        data = []
        for index in self.image_set_index:
            cur_data = self._process_keypoint_annotations(index)
            data.append(cur_data)
        self.num_instances = len(data)
        return data


    def _process_keypoint_annotations(self, index):
        """
        Extracting and processing annotations and metadata from the raw annotations file

        Args:
        -----
        index: integer
            Annotation id of the current ground truth to process
        """

        # loading annotations and metadata for index
        im_ann = self.coco.loadImgs(index)[0]
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # preprocessing ground truth
        bbox = objs[0]["bbox"]
        kpts_array = np.array(objs[0]['keypoints'])
        kpts = kpts_array.reshape(len(kpts_array)//3, 3)
        archdata_kpts_array = np.array(objs[0]['archdata_kpts'])
        archdata_kpts = archdata_kpts_array.reshape(len(archdata_kpts_array)//3, 3)
        center, scale = self._box2cs(*bbox)
        joints_vis = np.array([[k[-1], k[-1], 0] for k in kpts])

        # creating database ground truth instance
        data = {
            'image': im_ann["file_name"],
            'image_id': index,
            'original_image': im_ann["full_name"],
            'center': center,
            'scale': scale,
            'score': objs[0]["num_keypoints"],
            'kpts': kpts,
            'kpt_vis': joints_vis,
            'archdata_kpts': archdata_kpts,
            'character_name': objs[0]["character_name"]
        }

        return data


    def _process_bbox_annotations(self, index):
        """
        Extracting clean bounding box and class annotations from the raw annotations file

        Args:
        -----
        index: integer
            Annotation id of the current ground truth to process
        """

        im_ann = self.coco.loadImgs(index)[0]
        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        bboxes = []
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = x
            y1 = y
            x2 = x + w - 1
            y2 = y + h - 1
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                bboxes.append(obj["clean_bbox"])
        objs = valid_objs

        # obtaining the image targets (class, (x,y,w,h))
        # while keeping only the desired classes (person)
        targets = {
            "img_name": None,
            "img_path": None,
            "image_id": 0,
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": [],
            "arch_labels": [],
            "arch_labels_str": [],
        }

        for i, obj in enumerate(objs):
            class_id = obj['category_id']
            image_name = obj['img_name']
            image_path = os.path.join(self.data_path, obj["filename"])
            class_str = self.arch_labels_map[class_id]
            if(class_str not in self.filter):
                continue
            targets["img_name"] = image_name
            targets["img_path"] = image_path
            targets["boxes"].append(obj["clean_bbox"])
            targets["labels"].append(1)
            targets["area"].append(obj['area'])
            targets["iscrowd"].append(0)
            targets["arch_labels"].append(class_id)
            targets["arch_labels_str"].append(class_str)
            self.num_instances = self.num_instances + 1


        # formating and reshaping annotations
        targets["boxes"] = torch.Tensor(targets["boxes"]).float()
        targets["labels"] = torch.Tensor(targets["labels"]).float()
        targets["area"] = torch.Tensor(targets["area"])
        targets["iscrowd"] = torch.Tensor(targets["iscrowd"])

        return targets


    def _box2cs(self, x, y, w, h):
        """
        Converting a bounding box (top, left, width, height) to (center, scale)
        """
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0 / self.image_size[0], h * 1.0 / self.image_size[1]])
        # scale = np.array(
            # [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            # dtype=np.float32)
        # if center[0] != -1:
            # scale = scale * 1.25

        return center, scale


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_kpts, 3]
        :param joints_vis: [num_kpts, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_kpts, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        self.target_type = 'gaussian'

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_kpts,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_kpts):
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

        # if self.use_different_joints_weight:
            # target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

#
