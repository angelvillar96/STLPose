"""
Spliting the ArchDataset into a train and validation set. Saving all images and
annotations into the /data directory to load when training the EfficientDet.

@author: Angel Villar-Corrales 
"""

import os, shutil
import json
import argparse
from tqdm import tqdm

import numpy as np
from PIL import Image

from data import ArchDataset
from lib.utils import create_directory
from CONFIG import CONFIG


def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_size", help="Percentage of the data to save in for " \
                        "validation. Must be in range [0, 0.5]", default=0.2, type=float)
    parser.add_argument("--random_seed", help="Random seed used for the random " \
                        "permutation of the dataset", type=int, default=13)
    args = parser.parse_args()

    valid_size = args.valid_size
    random_seed = args.random_seed

    assert 0 <= valid_size <= 0.5, f"Validation size must be in range [0, 0.5]"

    return valid_size, random_seed


def split_dataset(valid_size=0.2, random_seed=13):
    """
    Logic for loading the arch-dataset and saving the respective data splits into
    train and validation
    """

    # loading datasets
    train_set = ArchDataset(task="pose_estimation", split_set="train",
                            valid_size=valid_size, create_split=True)
    valid_set = ArchDataset(task="pose_estimation", split_set="validation",
                            valid_size=valid_size, create_split=True)

    # creating directories
    dict_path = CONFIG["paths"]["dict_path"]
    split_dict = os.path.join(dict_path, "arch_data_pose_splits.json")

    if(os.path.exists(split_dict)):
        print(f"Split Datasets already exists. This process will overwrite previous results.")
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt != "y" and txt != "Y"):
            return

    # moving datasets
    train_ids = train_set.split_ids.tolist()
    print(train_ids[:10])
    valid_ids = valid_set.split_ids.tolist()
    print(valid_ids[:10])
    split = {
        "train": train_ids,
        "validation": valid_ids,
        "test": valid_ids
    }
    print("Saving...")
    with open(split_dict, "w") as file:
        json.dump(split, file)
    print("Saved! :)")
    return


if __name__ == "__main__":

    os.system("clear")
    valid_size, random_seed = process_arguments()

    split_dataset(valid_size=valid_size, random_seed=random_seed)

#
