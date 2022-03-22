"""
Displaying some of the skeleton instances saved in the retrieval database

@author: 
"""

import os
import argparse
from tqdm import tqdm

import numpy as np

from lib.arguments import process_experiment_directory_argument
from lib.logger import Logger, log_function, print_
from lib.pose_database import load_database
import lib.pose_parsing as pose_parsing
from lib.utils import create_directory, for_all_methods, load_experiment_parameters
from lib.visualizations import draw_pose, draw_skeleton
import CONSTANTS


def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--exp_directory", help="Path to the root of the " +\
                        "experiment folder", required=True, default="test_dir")
    parser.add_argument("--num_skeletons", help="Number of skeletons to display",
                        type=int, default=16)
    parser.add_argument("--database_split", help="Dataset split to process ['train', "\
                        "'eval', 'all']", default="eval")
    parser.add_argument("--shuffle", help="Boolean. If activated, skeletons are "\
                        "sampled at random", default="False")

    args = parser.parse_args()

    # making sure the experiment directory exists
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    num_skeletons = args.num_skeletons
    db_split = args.database_split
    assert db_split in ["train", "eval", "all"]
    shuffle = (args.shuffle == "True")

    return exp_directory, db_split, num_skeletons, shuffle


@log_function
def main(exp_directory, db_split="eval", num_skeletons=16, shuffle=True):
    """
    Main logic for the visualization of database skeletons:
        1.- Loading pickled database
        2.- Saving some skeletons into images

    Args:
    -----
    exp_directory: string
        path to the root directory of the experiment
    db_split: string
        name of the database split to process
    num_skeletons: integer
        number of skeletons to display and save
    shuffle: boolean
        If true, skeletons from the database are sampled at random
    """

    # macros
    pose_coco = CONSTANTS.SKELETON_HRNET
    pose_arch = CONSTANTS.SKELETON_SIMPLE

    # loading experiment parameters and relevant paths
    exp_data = load_experiment_parameters(exp_directory)
    dataset_name = exp_data["dataset"]["dataset_name"]
    skeletons_path = os.path.join(exp_directory, "plots", "db_skeletons")
    create_directory(skeletons_path)
    pose_parsing.SKELETON = pose_arch if(dataset_name == "arch_data") else pose_coco

    # loading databaset for the corresponding dataset and printing some skeletons
    database = load_database(db_name=dataset_name, db_split=db_split)
    for i, key in enumerate(database):
        print(database[key])
        if(i==5):
            break
    n_entries = len(database.keys())

    # sampling some entries
    if(shuffle):
        indices = np.random.randint(low=0, high=n_entries, size=num_skeletons)
    else:
        indices = np.arange(num_skeletons)

    all_keys = list(database.keys())
    empty_img = np.zeros((256,192,3))
    for i,idx in enumerate(tqdm(indices)):
        kpts = database[all_keys[idx]]["joints"].numpy()
        kpts[:,-1] = 1  # adding visibility
        kpts = [kpts[:, 1], kpts[:, 0], kpts[:, 2]]
        kpts = np.array(kpts).T
        savepath = os.path.join(skeletons_path, f"skeleton_{i+1}.png")
        draw_pose(img=empty_img, poses=[np.arange(19)], all_keypoints=kpts, savepath=savepath)

    return


if __name__ == "__main__":
    os.system("clear")

    # processing command line arguments
    exp_directory, db_split, num_skeletons, shuffle = process_arguments()

    # initializing logger
    logger = Logger(exp_directory)
    message = f"Initializing visualization of pose skeletons from database"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_info(message="Parameters:")
    logger.log_info(message="-----------")
    logger.log_info(message=f"    - exp_directory: {exp_directory}")
    logger.log_info(message=f"    - db_split: {db_split}")
    logger.log_info(message=f"    - num_skeletons: {num_skeletons}")
    logger.log_info(message=f"    - shuffle: {shuffle}")

    main(exp_directory, db_split, num_skeletons, shuffle)
    message = f"Pose skeletons displayed and saved successfully"
    logger.log_info(message=message)

#
