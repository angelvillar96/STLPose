"""
Evaluating the faster rcnn (or other model) for the task of person detection

EnhancePoseEstimation/src
@author: Angel Villar-Corrales 
"""

import os, pdb
import math
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel

from data.data_loaders import get_detection_dataset
import data.data_processing as data_processing
import lib.arguments as arguments
from lib.bounding_box import bbox_filtering, bbox_nms
import lib.model_setup as model_setup
import lib.inference as inference
import lib.metrics as metrics
import lib.loss as loss
import lib.utils as utils
from lib.visualizations import visualize_bbox
from lib.logger import Logger, log_function, print_
from lib.utils import for_all_methods, load_experiment_parameters
from CONFIG import CONFIG

from lib.detection_coco_utils import get_coco_api_from_dataset
from lib.detection_coco_eval import CocoEvaluator


@for_all_methods(log_function)
class DetectorEvaluator:
    """
    Class used for evaluating the FasterRCNN (or other model) for
    the task of person detection

    Args:
    -----
    exp_path: string
        path to the experiment directory
    checkpoint: string
        name of the checkpoit to load to resume training
    """

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, params=None):
        """
        Initializer of the detector evaluation object
        """

        self.exp_path = exp_path
        self.checkpoint = checkpoint
        self.params = params
        self.models_path = os.path.join(self.exp_path, "models", "detector")
        self.exp_data = load_experiment_parameters(exp_path)
        if(checkpoint is None):
            self.checkpoint_path = None
        else:
            self.checkpoint_path = os.path.join(self.models_path, self.checkpoint)

        if(dataset_name is not None):
            self.exp_data["dataset"]["dataset_name"] = dataset_name

        fname = f"test_detections_{self.exp_data['dataset']['dataset_name']}"
        self.plots_path = os.path.join(self.exp_path, "plots", fname)
        utils.create_directory(self.plots_path)

        self.class_ids = [1]
        self.num_classes = len(self.class_ids)
        self.bbox_thr = self.exp_data["evaluation"]["bbox_thr"]
        self.det_nms_thr = self.exp_data["evaluation"]["det_nms_thr"]

        return


    def load_detection_dataset(self):
        """
        Loading training and validation data-loaders. The data used corresponds to
        images from the (styled) COCO dataset. We only train for certain object classes,
        i.e., we only care about the ids specified by class_ids.
        """

        # updating alpha and styles if necesary
        if(self.params.alpha is not None):
            print_(f"'Alpha' parameter changing to {self.params.alpha}")
            self.exp_data["dataset"]["alpha"] = self.params.alpha
        if(self.params.styles is not None):
            print_(f"'Styles' parameter changing to {self.params.styles}")
            self.exp_data["dataset"]["styles"] = self.params.styles

        self.dataset_name = self.exp_data["dataset"]["dataset_name"]
        _, valid_loader  = get_detection_dataset(exp_data=self.exp_data, train=False,
                                                 validation=True, shuffle_train=True,
                                                 shuffle_valid=False, class_ids=self.class_ids)
        self.valid_loader = valid_loader

        self.coco = get_coco_api_from_dataset(self.valid_loader.dataset)
        self.iou_types = ["bbox"]

        return


    def load_detector_model(self):
        """
        Loading the pretrained person detector model
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_name = self.exp_data["model"]["detector_name"]
        model_type = self.exp_data["model"]["detector_type"]
        model = model_setup.setup_detector(model_name=model_name, model_type=model_type,
                                           pretrained=True, num_classes=self.num_classes)
        model.eval()

        if(model_name == "faster_rcnn"):
            model = DataParallel(model).to(self.device)

        # loading pretraining checkpoint if specified
        if(self.checkpoint is not None):
            print_(f"Loading checkpoint {self.checkpoint}")
            model = model_setup.load_checkpoint(self.checkpoint_path, model=model,
                                                only_model=True)

        if(model_name == "efficientdet"):
            model = DataParallel(model).to(self.device)

        self.model = model.to(self.device)

        return


    @torch.no_grad()
    def evaluate(self):
        """
        Evaluating the model on the dataset validation set

        Args:
        -----
        save: boolean
            If True, processed images are saved into the plots directory
        """

        self.model.eval()
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        self.n_img = 0

        for i, (imgs, metas_) in enumerate(tqdm(self.valid_loader)):

            imgs = torch.stack(imgs)
            metas = [{k: v for k, v in t.items()} for t in metas_]

            # prediciting bounding boxes
            imgs = imgs.to(self.device).float()
            outputs_ = self.model(imgs / 255)

            # mapping image ids with annotatons and outputs
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs_]
            res = {meta["image_id"]: output for meta, output in zip(metas, outputs)}

            self.coco_evaluator.update(res)

            # saving images if flag is set to True
            if(not self.params.save):
                continue
            # filtering boxes by threshold an applying NMS
            boxes, labels, scores = bbox_filtering(outputs_, filter_=1, thr=self.bbox_thr)
            boxes, labels, scores = bbox_nms(boxes, labels, scores, self.det_nms_thr)
            for i in range(imgs.shape[0]):
                savepath = os.path.join(self.plots_path, f"test_img_{self.n_img+1}.png")
                self.n_img = self.n_img + 1
                visualize_bbox(imgs[i,:].cpu().numpy().transpose(1,2,0) / 255,
                               boxes=boxes[i], labels=labels[i], scores=scores[i],
                               savepath=savepath, axis_off=True)

        # accumulate predictions from all images and computing evaluation metric
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.valid_stats = self.coco_evaluator.summarize()["bbox"].tolist()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                       'AR .75', 'AR (M)', 'AR (L)']
        print_("Validation Stats")
        print_(f"{stats_names}")
        print_(f"{self.valid_stats}")

        utils.save_evaluation_stats(stats=self.valid_stats,
                                    exp_path=self.exp_path,
                                    detector=True,
                                    checkpoint = self.checkpoint,
                                    dataset_name=self.dataset_name,
                                    alpha=self.exp_data["dataset"]["alpha"],
                                    styles=self.exp_data["dataset"]["styles"])
        return


if __name__ == "__main__":

    os.system("clear")
    exp_path, checkpoint, dataset_name, \
         params = arguments.get_directory_argument(get_checkpoint=True,
                                                   get_dataset=True)

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Initializing Faster-RCNN evaluation procedure"
    logger.log_info(message=message, message_type="new_exp")

    evaluator = DetectorEvaluator(exp_path=exp_path, checkpoint=checkpoint,
                                  dataset_name=dataset_name, params=params)
    evaluator.load_detection_dataset()
    evaluator.load_detector_model()
    evaluator.evaluate()

    logger.log_info(message=f"Evaluation of the Faster-RCNN finished successfully")


#
