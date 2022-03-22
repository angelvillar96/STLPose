"""
Loading one or two models and qualititatively observing where they perform better
or worse, and where their performance differes the most.

@author: 
"""

import os, json
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.nn import DataParallel

from data.data_loaders import get_detection_dataset, load_dataset
from lib.bounding_box import bbox_filtering, bbox_nms
from lib.detection_coco_utils import get_coco_api_from_dataset
from lib.detection_coco_eval import CocoEvaluator
from lib.logger import Logger, log_function, print_
from lib.utils import create_directory, load_experiment_parameters, for_all_methods
from lib.visualizations import visualize_bbox
from CONFIG import CONFIG
import data.data_processing as data_processing
import lib.metrics as metrics
import lib.model_setup as model_setup


def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument("--models", help="Relative paths to the model checkpoints to " +\
                        "load and compare. If more than one model is give, the paths " +\
                        "must be separated by comas. For example '--models path1,path2'",
                        required=True)
    parser.add_argument("--model_type", help="Type of models to compare: " +\
                        "['detection', 'pose']", default="detection")

    # directory parameters
    parser.add_argument("--output", help="Subdirectory of experiments/model_comparison " +\
                        "where the results will be stored")

    # data parameters
    parser.add_argument("--dataset", help="Dataset used for the evaluation", default="arch_data")
    parser.add_argument("--n_imgs", help="Number of images to evaluate", type=int, default=1e5)

    # functionality parameters
    parser.add_argument("--generate_imgs", help="If True, images with predictions are saved")
    parser.add_argument("--evaluate_map", help="If True, mAP is evaluated for each image")

    args = parser.parse_args()

    # enforcing correct values
    assert args.model_type in ["detection", "pose"], f"ERROR! model type {args.model_type} " +\
        "can only be one of the following ['detection', 'pose']"
    assert args.dataset in ["coco", "styled_coco", "arch_data", "red_black"]

    # checking that model parameter is correct
    models = args.models.split(",")
    # assert len(models) < 3, "ERROR! 'models' parameter can contain maximum 2 paths"
    exp_path = CONFIG["paths"]["experiments_path"]
    model_paths = []
    for model in models:
        model_path = os.path.join(exp_path, model)
        assert os.path.exists(model_path), f"ERROR! Model {model_path} does not exists"
        model_paths.append(model)
    args.models = model_paths

    # creating output directory
    out_path = os.path.join(CONFIG["paths"]["comparisons_path"], args.output)
    create_directory(out_path)
    args.output = out_path

    args.generate_imgs = (args.generate_imgs == "True") if args.generate_imgs != None else False
    args.evaluate_map = (args.evaluate_map == "True") if args.evaluate_map != None else False

    return args


@log_function
class ModelComparison:
    """
    Class that implements the model comparison pipeline
    """

    @log_function
    def __init__(self, params):
        """
        Intializer of the pipeline

        Args:
        -----
        params: Namespace
            Namespace containing the command line arguments
        """

        # command line arguments
        self.comparsion_path = params.output
        self.model_paths = params.models
        self.model_type = params.model_type
        self.params = params

        # paths and parameters
        self.models = []
        self.n_models = len(self.model_paths)
        self.exp_paths = [os.path.join(CONFIG["paths"]["experiments_path"],
                                       *m.split("/")[:2])
                          for m in self.model_paths
                         ]
        self.checkpoints = [m.split("/")[-1] for m in self.model_paths]
        self.results_file = os.path.join(self.comparsion_path, "results.json")

        # creating output directory for each model
        self.out_dirs = []
        for i in range(len(self.exp_paths)):
            exp_name = self.exp_paths[i].split("/")[-1]
            out_dir = os.path.join(self.comparsion_path, exp_name)
            self.out_dirs.append(out_dir)
            create_directory(out_dir)
            create_directory(path=out_dir, name="detections")
            create_directory(path=out_dir, name="poses")
            create_directory(path=out_dir, name="results")

        # loading exp_data from model_1 (we need some of the parameters later on)
        self.exp_data = load_experiment_parameters(self.exp_paths[0])

        return


    @log_function
    def load_data(self):
        """
        Loading comparison/evaluation dataset
        """

        self.exp_data["training"]["batch_size"] = 1
        self.exp_data["dataset"]["dataset_name"] = self.params.dataset

        if(self.model_type == "detection"):
            _, valid_loader  = get_detection_dataset(exp_data=self.exp_data, train=False,
                                                     validation=True, shuffle_train=False,
                                                     shuffle_valid=False, class_ids=[1])
        elif(self.model_type == "pose"):
            _, valid_loader = load_dataset(exp_data=self.exp_data, train=False,
                                           validation=True, shuffle_train=False,
                                           shuffle_valid=False)

        self.valid_loader = valid_loader

        return


    @log_function
    def load_model(self):
        """
        Loading the pretrained models that are going to be evaluated/compared
        """

        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = self.exp_data["model"]["detector_name"]
        model_type = self.exp_data["model"]["detector_type"]

        for i in range(self.n_models):
            # intializing corresponding model
            if(self.model_type == "detection"):
                checkpoint_path = os.path.join(self.exp_paths[i], "models",
                                               "detector", self.checkpoints[i])
                model = model_setup.setup_detector(model_name=model_name, model_type=model_type,
                                                   pretrained=True, num_classes=1)
            elif(self.model_type == "pose"):
                checkpoint_path = os.path.join(self.exp_path, "models", self.checkpoints[i])
                model = model_setup.load_model(self.exp_data, checkpoint=False)

            # loading pretrained model parameters
            self.model = DataParallel(model)
            print_(f"Loading checkpoint {self.checkpoints[i]}")

            self.model = model_setup.load_checkpoint(checkpoint_path, model=self.model,
                                                     only_model=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.models.append(model)

        return


    @log_function
    def inference(self):
        """
        Running inference through the entire evaluation set, storing the evaluation metrics
        """

        self.results = {f"model_{i}":[] for i in range(self.n_models)}
        self.results["ids"] = []

        if(self.model_type == "detection"):
            if(self.params.evaluate_map == True):
                self.coco = get_coco_api_from_dataset(self.valid_loader.dataset)
                self.iou_types = ["bbox"]
            self.detection_inference()
        elif(self.model_type == "pose"):
            print_("WARNING! Comparison of pose estimation models is not yet implemented...")
            exit()
            # self.pose_inference()

        # storing results
        with open(self.results_file, "w") as file:
            json.dump(self.results, file)

        return


    @log_function
    def detection_inference(self):
        """
        Running the inference through the detection models. Storing COCO-like mAP
        """

        for i, (imgs, metas) in enumerate(tqdm(self.valid_loader)):
            if(i >= self.params.n_imgs):
                break
            for j, cur_model in enumerate(self.models):

                img_name = metas["image_name"][0]
                # prediciting bounding boxes
                imgs = imgs.to(self.device).float()
                outputs_ = cur_model(imgs / 255)

                # computing and fetching mAP metric
                if(self.params.evaluate_map == True):
                    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs_]
                    res = {meta["image_id"].item(): output for meta, output in zip([metas], outputs)}
                    self._evaluate_bbox_dets(predictions=res, img_name=img_name)
                # displaying results and saving
                if(self.params.generate_imgs == True):
                    self._save_bbox_img(img=imgs, outputs=outputs_,
                                        img_name=img_name, idx=j)

        return


    def _evaluate_bbox_dets(self, img_name, predictions, idx=0):
        """
        Computing the COCO-evaluationg of the bounding box predictions

        Args:
        -----
        img_name: string
            name of the image being processed
        predictions: dictionary
            dict containing labels, scores and bbox coordinates for the predictions
        idx: integer
            idx of the model being evaluated
        """

        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        self.coco_evaluator.update(predictions)
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        valid_stats = self.coco_evaluator.summarize()["bbox"].tolist()
        cur_map = valid_stats[0]
        if(id not in self.results["ids"]):
            self.results["ids"].append(img_name.split(".")[0])
        self.results[f"model_{idx}"].append(cur_map)

        return


    def _save_bbox_img(self, img, outputs, img_name, idx=0, bbox_thr=0.7, nms_thr=0.5):
        """
        Saving an image overlayed with the detections

        Args:
        -----
        img: torch Tensor
            original image in which detzections will be overlayed
        outputs: torch tensor
            predictiobs of the detector model
        idx: integer
            idx of the model being evaluated
        """
        savepath = os.path.join(self.out_dirs[idx], "detections", f"{img_name}")
        boxes, labels, scores = bbox_filtering(outputs, filter_=1, thr=bbox_thr)
        boxes, labels, scores = bbox_nms(boxes, labels, scores, nms_thr)
        visualize_bbox(img[0,:].cpu().numpy().transpose(1,2,0) / 255,
                       boxes=boxes, labels=labels, scores=scores,
                       savepath=savepath, axis_off=True)

        return


if __name__ == "__main__":
    os.system("clear")

    # processign arguments and initializing logger
    args = process_arguments()
    logger = Logger(args.output)
    message = f"Initializing model comparison"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_params(params=vars(args))

    # model comparison pipeline
    comparison = ModelComparison(params=args)
    comparison.load_data()
    comparison.load_model()
    comparison.inference()

#
