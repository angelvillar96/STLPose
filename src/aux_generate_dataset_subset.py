"""
Script for testing the functionality of dataloaders. Checking that pairs
of (image, *target) are created correctly

EnhancePoseEstimation/src
@author: 
"""

import os

import numpy as np
from matplotlib import pyplot as plt

import data
import lib.arguments as arguments
import lib.utils as utils
import lib.visualizations as visualizations
import lib.pose_parsing as pose_parsing
from lib.pose_parsing import get_final_preds_hrnet, create_pose_entries
import CONSTANTS

def main():
    """
    Main orquestrator for loading the data, calling first batch, creating and
    saving the plots
    """

    pose_parsing.SKELETON = CONSTANTS.SKELETON_HRNET

    # relevant paths and data
    exp_directory = arguments.get_directory_argument()
    plots_path = os.path.join(exp_directory, "plots")
    exp_data = utils.load_experiment_parameters(exp_directory)
    exp_data["training"]["batch_size"] = 1

    # saving some pose estimation examples
    fig, ax = plt.subplots(2,3)
    savepath = os.path.join(plots_path, "dataset_subset_pose_estimation.png")
    _, valid_loader = data.load_dataset(exp_data=exp_data, train=False,
                                        validation=True, shuffle_valid=True)
    for i, (img, target, target_weight, metadata) in enumerate(valid_loader):
        row, col = i // 3, i % 3
        if(i==6):
            break
        kpts = metadata["joints"][0,:].numpy()
        kpts[:,-1] = metadata["joints_vis"][0][:,0]
        kpts = np.array([kpts[:,1], kpts[:,0], kpts[:,-1]]).T
        visualizations.draw_pose(img[0,:].numpy(), poses=[np.arange(19)],
                                 all_keypoints=kpts, preprocess=True,
                                 ax=ax[row,col], axis_off=True)
    plt.savefig(savepath)

    # saving some person detection examples
    fig, ax = plt.subplots(2,3)
    savepath = os.path.join(plots_path, "dataset_subset_person_detection.png")
    _, valid_loader  = data.get_detection_dataset(exp_data=exp_data, train=False,
                                                  validation=True, shuffle_valid=True)
    for i, (img, metas) in enumerate(valid_loader):
        row, col = i // 3, i % 3
        if(i==6):
            break
        boxes = metas["targets"]["boxes"][0,:]
        labels = metas["targets"]["labels"][0,:]
        visualizations.visualize_bbox(img=img[0,:].numpy(), boxes=boxes, labels=labels,
                                      preprocess=True, ax=ax[row,col])
    plt.savefig(savepath)

    return


if __name__ == "__main__":
    os.system("clear")
    main()

#
