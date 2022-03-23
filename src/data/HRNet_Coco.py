"""
(Styled-) COCO dataset for training the HRNet model

Adapted from: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
from collections import OrderedDict
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np
import torch

from data.JointsDataset import JointsDataset
from lib.nms import oks_nms
import CONSTANTS


class HRNetCoco(JointsDataset):
    """
    Class for loading and processing of the COCO dataset.
    Inherits from the JointsDataset. See JointsDataset for:
        - __getitem__()
        - get_target()

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
    alpha: string
        identifier of the amount of styled transfered into Styled-COCO ['random', '1.0', '0.5']
    transform: Transforms
        transforms to apply to the images  (e.g., toTensor, Normalize or Resize)
    """

    def __init__(self, exp_data, root, img_path, labels_path, is_train, is_styled=False,
                 alpha=None, styles=None, perceptual_loss_dict=None, transform=None):
        """ Module initializer """
        super().__init__(exp_data=exp_data, root=root, img_path=img_path,
                         labels_path=labels_path, is_train=is_train,
                         perceptual_loss_dict=perceptual_loss_dict, transform=transform)

        self.nms_thre = exp_data["evaluation"]["nms_thr"]
        self.image_thre = exp_data["evaluation"]["img_thr"]
        self.oks_thre = exp_data["evaluation"]["oks_thr"]
        self.in_vis_thr = exp_data["evaluation"]["in_vis_thr"]
        self.bbox_file = os.path.join(
                self.root,
                "person_detection_results",
                "COCO_val2017_detections_AP_H_56_person.json"
            )
        self.use_gt_bbox = exp_data["evaluation"]["use_gt_bbox"]
        self.image_width = 192
        self.image_height = 256
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.is_train = is_train
        self.is_styled = is_styled
        self.alpha = alpha
        self.styles = styles

        if (self.is_styled):
            self.styled_image_names = os.listdir(img_path)
            self.mapping_dict = self._load_mapping_dict()

        self.annotations_file = self.labels_path
        self.coco = COCO(self.labels_path)

        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
                [(self._class_to_coco_ind[cls], self._class_to_ind[cls])for cls in self.classes[1:]]
            )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)

        self.num_joints = len(CONSTANTS.COCO_MAP_HRNET)
        self.flip_pairs = CONSTANTS.FLIP_PAIRS
        self.parent_ids = None
        self.upper_body_ids = CONSTANTS.UPPER_BODY_IDS
        self.lower_body_ids = CONSTANTS.LOWER_BODY_IDS

        self.joints_weight = np.array([
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
            ], dtype=np.float32).reshape((self.num_joints, 1))

        self.db = self._get_db()
        return

    def _load_image_set_index(self):
        """
        Obtaining image_ids from the COCO images
        """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        """
        Obtaining the bounding box annotations, either from the COCO annotations or
        from the ones saved by the object detector
        """
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """
        ground truth bbox and keypoints
        """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

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
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            image_path = self.image_path_from_index(index)
            image_name = os.path.basename(image_path)

            # obtaining current value of alpha
            if(self.alpha == "random" and "alpha" in image_name):
                alpha = image_name.split("alpha_")[-1].split(".jpg")[0]
                alpha = float(alpha)
            else:
                alpha = self.alpha if self.alpha is not None else 0
            # getting path of original image if in styled model
            if(self.is_styled):
                original_image_path = self.original_image_path_from_index(index)
            else:
                original_image_path = image_path

            rec.append({
                'image': image_path,
                'original_image': original_image_path,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'alpha': alpha
            })
        return rec

    def _box2cs(self, box):
        """ Reparameterizing BBOX from (top, left, width, height) to (center, scale)"""
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """ Converting a bounding box (top, left, width, height) to (center, scale) """
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

            # scale = np.array([w * 1.0 / self.image_size[0], h * 1.0 / self.image_size[1]])
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def _load_mapping_dict(self):
        """
        Loading the mapping dictionary that assignas COCO image names to their
        corresponding styled counterpart
        """

        alpha = self.alpha
        style = self.styles
        if self.is_train:
            cur_dict = f"train_dict_style_{style}_alpha_{alpha}.json"
        else:
            cur_dict = f"valid_dict_style_{style}_alpha_{alpha}.json"
        mapping_dict_path = os.path.join(self.root, "mapping_dicts", cur_dict)
        print(f"Loading {mapping_dict_path}...")

        if(not os.path.exists(mapping_dict_path)):
            raise FileNotFoundError(
                f"Dictionary mapping COCO to Styled_COCO-{alpha}-{style} does not exists."
                "Run 'aux_styled_coco_preload' to generate the dictionaries."
            )

        with open(mapping_dict_path) as f:
            mapping_dict = json.load(f)
        return mapping_dict

    def _get_styled_image_given_original(self, original_name):
        """ Fetching the name of the styled image given the name of the original one """
        original_name = '%012d' % float(original_name)
        # if(original_name not in self.mapping_dict.keys()):
        #     return None
        cur_styled_img_name = self.mapping_dict[original_name]
        return cur_styled_img_name

    def image_path_from_index(self, index):
        """ Obtaining an image path given the image ID """
        if(self.is_styled):
            file_name = self._get_styled_image_given_original(str(index))
            if(file_name is None):
                return None
            image_path = os.path.join(self.img_path, file_name)
        else:
            file_name = '%012d.jpg' % index
            image_path = os.path.join(self.original_image_path, file_name)
        return image_path

    def original_image_path_from_index(self, index):
        """ Obtaining the original COCO image path given the ID """
        file_name = '%012d.jpg' % index
        image_path = os.path.join(self.original_image_path, file_name)
        return image_path

    def get_name_given_id(self, index):
        """ Obtaining an image name given the image ID """
        file_name = self.db[index]['original_image'].split("/")[-1]
        if(self.is_styled):
            file_name = self._get_styled_image_given_original(file_name[:-4])
        return file_name

    def _load_coco_person_detection_results(self):
        """ Loading bounding box annotations extracted and exported by a human detector """
        if(not os.path.exists(self.bbox_file)):
            raise NameError(f"ERROR: Bounding Box annotation file: {self.bbox_file} does not exist...")

        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            return None

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            original_img_name = self.original_image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1
            center, scale = self._box2cs(box)

            # TODO: missing loading annotations
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'original_image': original_img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        return kpt_db

    def get_all_samples_given_name(self, name):
        """ Obtaining all persons, bboxes and annotations given an image name """
        inputs, targets, weights, metas = [], [], [], []
        for idx, db_rec in enumerate(self.db):
            image_file = db_rec['image'].split("/")[-1]

            if(image_file != name):
                continue

            input, target, target_weight, meta = self.__getitem__(idx)
            inputs.append(input)
            targets.append(target)
            weights.append(target_weight)
            metas.append(meta)

        # concatenating inputs and targets and merging dictionaries
        inputs = torch.Tensor([i.numpy() for i in inputs])
        targets = torch.Tensor([t.numpy() for t in targets])
        weights = torch.Tensor([w.numpy() for w in weights])
        metadata = {}
        for key in metas[0]:
            metadata[key] = []
        for meta in metas:
            for key in meta:
                metadata[key].append(meta[key])
        for key in ['joints', 'joints_vis', 'center', 'scale', 'rotation', 'score']:
            metadata[key] = torch.Tensor(metadata[key])

        return inputs, targets, weights, metadata

    #############################
    # TODO
    # MIGRATE THE METHODS BELOW THIS LINE TO LIBRARIES
    #############################

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                assert False, 'Fail to make {}'.format(res_folder)

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thr = self.in_vis_thr
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

#
