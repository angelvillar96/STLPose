"""
Extracting the pose skeletons from the arch data using a pretrained model
to create our estimated version of the ArchData dataset

@author: 
"""

import os
import pickle
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision.transforms as transforms

from lib.arguments import get_directory_argument
from lib.logger import Logger, log_function, print_
from lib.model_setup import load_checkpoint, load_model
from lib.pose_parsing import get_final_preds_hrnet, get_max_preds_hrnet, create_pose_entries
from lib.utils import create_directory, for_all_methods, load_experiment_parameters
from lib.visualizations import draw_pose, visualize_image
from CONFIG import CONFIG
import CONSTANTS
import data
import data.data_processing as data_processing
import lib.pose_parsing as pose_parsing
import lib.inference as inference


@for_all_methods(log_function)
class ArchDataExtractor:
    """
    Class to extract bounding box and keypoint annotations for all images
    from the ArchData dataset

    Args:
    -----
    exp_path: string
        path to the experiment directory
    person_det_checkpoint, keypoint_det_checkpoint: string
        name of the files containing the desired state dictionaries for the person
        and keypoint detection models
    """

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, params=None):
        """
        Initializer of the ArchData extractor object
        """

        self.exp_path = exp_path
        self.checkpoint = checkpoint
        self.params = params if params is not None else {}
        self.exp_data = load_experiment_parameters(exp_path)
        if(dataset_name is not None):
            self.exp_data["dataset"]["dataset_name"] = dataset_name
        self.dataset_name = self.exp_data["dataset"]["dataset_name"]

        # model and processing parameters
        self.kpt_thr = 0.1
        pose_parsing.SKELETON = CONSTANTS.SKELETON_HRNET
        data_processing.TO_COCO_MAP = CONSTANTS.COCO_MAP_HRNET

        # defining and creating directories to save the results
        plots_path = os.path.join(self.exp_path, "plots")
        create_directory(plots_path)
        self.results_path = os.path.join(plots_path, "final_result_imgs")
        create_directory(self.results_path)

        return


    def load_dataset(self):
        """
        Loading the ArchData dataset and fitting a data loader
        """

        self.exp_data["training"]["batch_size"] = 1
        _, self.valid_loader = data.load_dataset(exp_data=self.exp_data, train=False,
                                                 validation=True, shuffle_train=False,
                                                 shuffle_valid=False)
        self.image_size = self.exp_data["dataset"]["image_size"]

        return


    def load_models(self):
        """
        Loading pretrained models for person detection as well as keypoint detection
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setting up pretrained keypoint detector
        pose_model = load_model(self.exp_data, checkpoint=self.checkpoint)
        self.pose_model = DataParallel(pose_model)
        if(self.checkpoint is not None):
            print_(f"Loading checkpoint {self.checkpoint}")
            checkpoint_path = os.path.join(self.exp_path, "models",
                                          self.checkpoint)
            self.pose_model = load_checkpoint(checkpoint_path,
                                              model=self.pose_model,
                                              only_model=True)
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()
        self.model_name = self.exp_data["model"]["model_name"]

        return

    @torch.no_grad()
    def extract_retrieval_dataset(self):
        """
        Iterating over all images from the ArchData dataset, extracting the results
        from person detection and keypoint estimation. The results are saved in a json
        file to then use for retrieval database purposes
        """

        self.retrieval_db = {}
        rescale_dets = lambda dets: F.interpolate(dets, (256, 192),
                                                  mode="bilinear",
                                                  align_corners=True)

        for i, (imgs, _, _, metadata) in enumerate(tqdm(self.valid_loader)):

            imgs = imgs.to(self.device).float()
            centers = metadata["center"].numpy()
            scales = metadata["scale"].numpy()
            img_name = metadata["image"][0]

            keypoint_dets = inference.forward_pass(model=self.pose_model, img=imgs,
                                            model_name=self.model_name,
                                            device=self.device, flip=True)
            scaled_dets = rescale_dets(keypoint_dets.clone())

            # extracting keypoint coordinates from estimated heatmaps
            keypoint_coords, max_vals = get_max_preds_hrnet(scaled_dets.cpu().numpy())
            indep_pose_entries, indep_all_keypoints = create_pose_entries(keypoint_coords,
                                                                          max_vals,
                                                                          thr=self.kpt_thr)
            # arranging keypoint to visualize the skeletons
            indep_all_keypoints = [indep_all_keypoints[:, 1], indep_all_keypoints[:, 0],
                                   indep_all_keypoints[:, 2], indep_all_keypoints[:, 3]]
            indep_all_keypoints = np.array(indep_all_keypoints).T

            # creating current pose vector with shape (17,3): 17 kpts, (x,y,vis)
            # pose_kpts = np.array([indep_all_keypoints[int(idx)]
                                 # for idx in indep_pose_entries[j][:-2]])
            pose_kpts = np.array((indep_all_keypoints[:,1], indep_all_keypoints[:,0],
                                  indep_all_keypoints[:,-1])).T
            # saving current pose in retrieval database
            cur_data = {
                "img": img_name,
                "joints": torch.Tensor(pose_kpts).float(),
                "center": torch.Tensor(centers).float(),
                "scale": torch.Tensor(scales).float(),
                "character_name": metadata['character_name'][0]
            }
            new_key = f"img_{len(self.retrieval_db.keys())}"
            self.retrieval_db[new_key] = cur_data


            # converting keypoint locations from detection to image coordinates
            if(self.params.save == True):
                poses, kpts = pose_parsing.create_pose_from_outputs(dets=keypoint_dets,
                                                                    keypoint_thr=self.kpt_thr)

                draw_pose(img=imgs[0,:].cpu().numpy(), poses=[poses[0]],
                          all_keypoints=kpts, preprocess=True, axis_off=True,
                          savepath=os.path.join(self.results_path, img_name))

        return


    def save_retrieval_db(self):
        """
        Saving the retrieval db into a pickle file
        """

        create_directory(CONFIG["paths"]["database_path"])
        fname = f"database_{self.dataset_name}_{os.path.basename(self.exp_path)}_eval.pkl"
        database_path = os.path.join(CONFIG["paths"]["database_path"], fname)
        with open(database_path, "wb") as file:
            pickle.dump(self.retrieval_db, file)

        return


if __name__ == "__main__":
    os.system("clear")
    exp_path, checkpoint, dataset,\
        params = get_directory_argument(get_checkpoint=True, get_dataset=True)

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Starting to extract ArchData retrieval dataset."
    logger.log_info(message=message, message_type="new_exp")

    extractor = ArchDataExtractor(exp_path=exp_path, checkpoint=checkpoint,
                                  dataset_name=dataset, params=params)
    extractor.load_dataset()
    extractor.load_models()
    extractor.extract_retrieval_dataset()
    extractor.save_retrieval_db()
    logger.log_info(message=f"Dataset extracted successfully.")

#
