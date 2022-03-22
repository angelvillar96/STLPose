"""
(Styled-) COCO dataset for training an object detection model

EnhancePoseEstimation/src/data
@author: 
"""

from collections import defaultdict
from collections import OrderedDict
import os, pdb
import copy
import json
import sys

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from CONFIG import CONFIG

class DetectionCoco(Dataset):
    """
    Class for loading and processing of the COCO dataset for object/person detection.

    Args:
    -----
    exp_data: dictionary
        parameters from the current experiment
    root, img_path, labels_path: string
        path to the root data, image and annotations directories respectively
    is_train: boolean
        selects training and validation mode
    is_styled: boolean
        if True, searches for styled data rather than original
    class_ids: list of integers
        list containing the ids of the classes to detect. By defaul [1] (person class)
    transform: Transforms
        transforms to apply to the images  (e.g., toTensor, Normalize or Resize)
    """

    def __init__(self, exp_data, root, img_path, labels_path, is_train=False,
                 is_styled=False, class_ids=[1], resizer=None, perceptual_loss_dict=None,
                 alpha=None, styles=None, transform=None):

        # object parameters
        self.exp_data = exp_data
        self.root = root
        self.img_path = img_path
        self.labels_path = labels_path
        self.is_train = is_train
        self.is_styled = is_styled
        self.class_ids = class_ids
        self.resizer = resizer
        self.transform = transform
        self.perceptual_loss_dict = perceptual_loss_dict
        self.alpha = alpha
        self.styles = styles

        if (self.is_styled):
            self.styled_image_names = os.listdir(img_path)
            self.mapping_dict = self._load_mapping_dict()

        # initializing data with COCO-API and getting image name
        self.coco = COCO(self.labels_path)
        self.image_set_index = self._load_image_set_index()

        # dealing with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [ (self._class_to_coco_ind[cls], self._class_to_ind[cls])
              for cls in self.classes[1:] ]
        )

        # loading data into data structure
        self.data = self._load_coco_keypoint_annotations()
        self.num_images = len(self.data)

        return


    def __len__(self):
        """Obtaining the total number of samples in the dataset"""
        return self.num_images


    def __getitem__(self, idx):
        """
        Obtaining the data (image, annotations and metadata) given a dataset idx
        """

        data = copy.deepcopy(self.data[idx])

        image_name = data['image_name']
        original_image_name = data['original_image_name']
        targets = data["targets"]
        image_id = data["image_id"]
        image_file = os.path.join(self.img_path, image_name)

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        original_image_size = data_numpy.shape
        if self.resizer is not None:
            data_numpy, targets, scale = self.resizer(data_numpy, targets)
        input = data_numpy
        if self.transform is not None:
            input = self.transform(input)

        # obtaining precomputed perceptual_loss
        perceptual_loss = None
        if self.perceptual_loss_dict:
            perceptual_loss = self.perceptual_loss_dict[image_name]

        meta = {
            'image_name': image_name,
            'original_image_name': original_image_name,
            "targets": targets,
            "image_id": image_id,
            "scale": scale,
            "original_size": original_image_size[:2],
            "perceptual_loss": perceptual_loss
        }

        return input, meta


    ##################################################################################
    # TODO
    # Some of the methods below are common to detection and pose-estimation datasets
    # maybe we should move them to a data_utils.py library to avoid code redundancies
    #################################################################################


    def _load_coco_keypoint_annotations(self):
        """
        ground truth bbox and keypoints
        """
        data = []
        for index in self.image_set_index:
            targets = self._load_coco_keypoint_annotation_kernal(index)
            targets["image_id"] = index
            cur_data = {}
            cur_data["image_name"] = self._image_name_from_index(index)
            cur_data["original_image_name"] = '%012d.jpg' % index
            cur_data["targets"] = targets
            cur_data["image_id"] = index
            # if the image does not contain any bounding box, we skip it
            if( len(cur_data["targets"]["labels"]) == 0 or cur_data["image_name"] is None):
                continue
            data.append(cur_data)
        return data


    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        Args:
        -----
        index: integer
            coco image id

        Returns:
        --------
        db entry: list of dictionaries
            annotations corresponding to the image whose ID was given as argument
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        bboxes = []
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                bboxes.append(obj["clean_bbox"])
        objs = valid_objs

        # obtaining the image targets (class, (x,y,w,h))
        # while keeping only the desired classes (person)
        targets = {
            "image_id": "",
            "boxes": [],
            "labels": [],
            "area": [],
            "iscrowd": []
        }

        for i, obj in enumerate(objs):
            class_ = self._coco_ind_to_class_ind[obj['category_id']]
            if( int(class_) not in self.class_ids):
                continue
            targets["boxes"].append(obj["clean_bbox"])
            targets["labels"].append(class_)
            targets["area"].append(obj['area'])
            targets["iscrowd"].append(0)

        # formating and reshaping annotations
        targets["boxes"] = torch.Tensor(targets["boxes"]).float()
        targets["labels"] = torch.Tensor(targets["labels"]).float()  #.long()
        targets["area"] = torch.Tensor(targets["area"])
        targets["iscrowd"] = torch.Tensor(targets["iscrowd"])#.type(torch.uint8)

        return targets


    def _load_image_set_index(self):
        """
        Obtaining image_ids from the COCO images
        """
        image_ids = self.coco.getImgIds()
        return image_ids


    def _load_mapping_dict(self):
        """
        Loading the mapping dictionary that assignas COCO image names to the
        corresponding styled counterpart
        """

        alpha = self.alpha
        style = self.styles
        if self.is_train:
            cur_dict = f"train_dict_style_{style}_alpha_{alpha}.json"
        else:
            cur_dict = f"valid_dict_style_{style}_alpha_{alpha}.json"
        mapping_dict_path = os.path.join(CONFIG["paths"]["dict_path"], cur_dict)
        print(f"Loading {mapping_dict_path}...")

        if( not os.path.exists(mapping_dict_path) ):
            assert False, f"Dictionary mapping COCO to Styled_COCO-{alpha}-{style} does " +\
                "not exists.\n Run 'aux_styled_coco_preload' to generate the dictionaries."

        with open(mapping_dict_path) as f:
            mapping_dict = json.load(f)

            return mapping_dict


    def _image_name_from_index(self, index):
        """
        Obtaining the original or styled COCO image name given the ID
        """

        image_name = '%012d.jpg' % index
        if(self.is_styled):
            image_name = self._get_styled_image_given_original(image_name[:-4])

        return image_name


    def _get_styled_image_given_original(self, original_name):
        """
        Fetching the name of the styled image given the name of the original one
        """

        original_name = '%012d' % float(original_name)
        # if(original_name not in self.mapping_dict.keys()):
            # return None
        cur_styled_img_name = self.mapping_dict[original_name]

        return cur_styled_img_name


#
