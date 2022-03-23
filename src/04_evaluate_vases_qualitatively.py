"""
Qualitative evaluation of the vase-data subset. The person instances are detected
and the poses estimated for each of the images. The results are then stored in the
plots directory of the experiment.

@author: Angel Villar-Corrales 
"""

import os
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision.transforms as transforms

import data.data_processing as data_processing
from data.data_loaders import get_vase_subset
from lib.arguments import process_experiment_directory_argument, process_checkpoint
from lib.bounding_box import bbox_filtering
from lib.logger import Logger, log_function, print_
from lib.model_setup import load_checkpoint, load_model, setup_detector
from lib.pose_parsing import get_final_preds_hrnet, get_max_preds_hrnet, create_pose_entries
import lib.pose_parsing as pose_parsing
from lib.transforms import TransformDetection
from lib.utils import create_directory, for_all_methods, load_experiment_parameters
from lib.visualizations import visualize_bbox, draw_pose
import CONSTANTS


def process_arguments():
    """
    Processing command line argumetns

    Returns:
    --------
    args: namespace
        namespace containing the values of the command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # experiment arguments
    parser.add_argument("-d", "--exp_directory", help="Name of the experiment to " +\
                        "consider for processing", required=True, default="test_dir")

    # model arguments
    parser.add_argument("--checkpoint_detector", help="Name of the checkpoint file "+\
                        "to load for the person detector model", required=True)
    parser.add_argument("--checkpoint_pose", help="Name of the checkpoint file to " +\
                        "to load for the pose estimator model", required=True)

    # nms thresholds for detection and pose estimation
    parser.add_argument("--detector_thr", help="Score threshold to consider a detection.",
                        type=float, default=0.8)
    parser.add_argument("--keypoint_thr", help="Score threshold to consider a keypoint.",
                        type=float, default=0.2)

    args = parser.parse_args()

    # enforcing correct values
    assert 0 <= args.detector_thr <= 1
    assert 0 <= args.keypoint_thr <= 1

    # making sure the experiment directory exists
    args.exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint_detector = process_checkpoint(args.checkpoint_detector, args.exp_directory)
    args.checkpoint_pose = process_checkpoint(args.checkpoint_pose, args.exp_directory)

    return args


@for_all_methods(log_function)
class VaseEvaluator:
    """"
    Class for evaluating the subset of vase data qualitatively. Each image is processed
    and the results are saved into the plots/qualitative_vases directory

    Args:
    -----
    exp_directory: string
        root directory of the experiment
    det_checkpoint, pose_checkpoint: string
        names of the pretrained models to load. None correspond to the un-tuned models
    detector_thr, keypoint_thr: float [0, 1]
        threshold for considering a person detection and a keypoint detection valid
    """

    def __init__(self, exp_directory, checkpoint_detector=None, checkpoint_pose=None,
                 detector_thr=0.7, keypoint_thr=0.1):
        """
        Initializer of the evaluator object
        """

        # generating relevant paths and loading experiment data
        self.exp_directory = exp_directory
        self.exp_data = load_experiment_parameters(exp_directory)
        self.models_path = os.path.join(exp_directory, "models")
        self.detector_path = os.path.join(exp_directory, "models", "detector")
        img_size = self.exp_data["dataset"]["image_size"]
        dir_name = f"qualitative_vases_DetCheckpoint_{checkpoint_detector}_"\
                   f"PoseCheckpoint_{checkpoint_pose}_DetThr_{detector_thr}_"\
                   f"PoseThr_{keypoint_thr}_img-size_{img_size}"
        self.plots_path = os.path.join(exp_directory, "plots", dir_name)
        self.det_plots_path = os.path.join(self.plots_path, "detections")
        create_directory(self.det_plots_path)
        self.pose_plots_path = os.path.join(self.plots_path, "poses")
        create_directory(self.pose_plots_path)
        self.instances_plots_path = os.path.join(self.plots_path, "instances")
        create_directory(self.instances_plots_path)


        if(checkpoint_detector is not None):
            self.det_checkpoint = os.path.join(self.detector_path, checkpoint_detector)
        else:
            self.det_checkpoint = None
        if(checkpoint_pose is not None):
            self.pose_checkpoint = os.path.join(self.models_path, checkpoint_pose)
        else:
            self.pose_checkpoint = None
        self.detector_thr = detector_thr
        self.keypoint_thr = keypoint_thr

        # macros
        pose_parsing.SKELETON = CONSTANTS.SKELETON_HRNET
        data_processing.TO_COCO_MAP = CONSTANTS.COCO_MAP_HRNET
        data_processing.SKIP_NECK = False

        return


    def load_vase_subset(self, verbose=0):
        """
        Obtaining the image paths to load the vase dataset.
        TODO: Might need change once we have the annotations

        Args:
        -----
        verbose: integer
            verbosity level
        """

        self.images  = get_vase_subset(img_size=self.exp_data["dataset"]["image_size"])
        self.n_imgs = len(self.images)
        if(verbose > 0):
            print_(f"Loaded {self.n_imgs} images from the Greek Vase subset")

        return


    def setup_models(self):
        """
        Loading models for person detections and for pose estimation using the
        pretrained checkpoints specified
        """

        # seting up GPU
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # loading person detector
        detector = setup_detector()
        detector = DataParallel(detector).to(self.device)
        detector = load_checkpoint(checkpoint_path=self.det_checkpoint,
                                   model=detector, only_model=True)
        self.detector = detector.eval()

        # loading keypoint detector
        hrnet = load_model(self.exp_data)
        hrnet = DataParallel(hrnet).to(self.device)
        hrnet = load_checkpoint(checkpoint_path=self.pose_checkpoint,
                                model=hrnet, only_model=True, map_cpu=True)
        self.hrnet = hrnet.eval()

        return


    @torch.no_grad()
    def qualitative_comparison(self):
        """
        Processing all images from the evaluation subset and storing the results
        in the corresponding plots directories
        """

        # online transformations for image processing
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        get_detections = TransformDetection(det_width=192, det_height=256)

        # iterating all subset images
        for i, img in enumerate(tqdm(self.images)):

            # preprocessing
            img_name = f"test_img_{i+1}.png"
            cur_img = torch.Tensor(img[np.newaxis,:])

            # person detection
            outputs = self.detector(cur_img / 255)
            boxes, labels, scores = bbox_filtering(outputs, filter_=1,
                                                   thr=self.detector_thr)

            # saving detections image
            savepath = os.path.join(self.det_plots_path, img_name)
            visualize_bbox(cur_img[0,:].cpu().numpy().transpose(1,2,0) / 255,
                           boxes=boxes[0], labels=labels[0], axis_off=True,
                           scores=scores[0], savepath=savepath)

            # preprocessing detections
            img_extract = cur_img[0,:].numpy().transpose(1,2,0)
            dets, centers, scales = get_detections(img=img_extract, list_coords=boxes[0])
            n_dets = dets.shape[0]
            if(n_dets == 0):
                continue
            normed_dets = [normalize(torch.Tensor(det/255)).numpy() for det in dets]

            # keypoint detection
            keypoint_dets = self.hrnet(torch.Tensor(normed_dets).float())
            scaled_dets = F.interpolate(keypoint_dets.clone(), (256, 192),
                                        mode="bilinear", align_corners=True)

            # pose parsing for person instances independently
            keypoint_coords, max_vals = get_max_preds_hrnet(scaled_dets.cpu().numpy())
            indep_pose_entries, indep_all_keypoints = \
                create_pose_entries(keypoint_coords, max_vals, thr=self.keypoint_thr)
            indep_all_keypoints = [indep_all_keypoints[:, 1], indep_all_keypoints[:, 0],
                                   indep_all_keypoints[:, 2], indep_all_keypoints[:, 3]]
            indep_all_keypoints = np.array(indep_all_keypoints).T
            for j,det in enumerate(dets):
                det_name = f"{img_name[:-4]}_det_{j+1}{img_name[-4:]}"
                savepath = os.path.join(self.instances_plots_path, det_name)
                draw_pose(det, [indep_pose_entries[j]], indep_all_keypoints,
                          preprocess=True, savepath=savepath, axis_off=True)

            # pose parsing for full image
            keypoints, max_vals, _ = get_final_preds_hrnet(keypoint_dets.cpu().numpy(),
                                                           centers, scales)
            pose_entries, all_keypoints =\
                create_pose_entries(keypoints, max_vals, thr=self.keypoint_thr)
            all_keypoints = [all_keypoints[:, 1], all_keypoints[:, 0],
                             all_keypoints[:, 2], all_keypoints[:, 3]]
            all_keypoints = np.array(all_keypoints).T
            savepath = os.path.join(self.pose_plots_path, img_name)
            draw_pose(cur_img[0,:].cpu().numpy().transpose(1,2,0) / 255,
                      pose_entries, all_keypoints, savepath=savepath, axis_off=True)

        return


if __name__ == "__main__":
    os.system("clear")

    # processing command line arguments and initializing logger
    args = process_arguments()
    logger = Logger(args.exp_directory)
    message = f"Initializing vase-subset qualitative evaluation"
    logger.log_info(message=message, message_type="new_exp")

    # evaluation pipeline
    evaluator = VaseEvaluator(**vars(args))
    evaluator.load_vase_subset(verbose=1)
    evaluator.setup_models()
    evaluator.qualitative_comparison()

    message = f"Qualitative evaluation completed successfully"
    logger.log_info(message=message)

#
