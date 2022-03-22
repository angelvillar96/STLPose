"""
Preprocessing a certain dataset (e.g. Styled COCO) to set as database in the
pose-based retrieval demonstrator. The preprocessing steps include saving bounding
box and keypoint annotations

@author: 
"""

import os
import argparse
from tqdm import tqdm
import pickle

import numpy as np
import torch

from data.data_loaders import load_dataset
from lib.utils import load_experiment_parameters, timestamp
from CONFIG import CONFIG


def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Dataset to preprocess ['coco', "\
                        "'styled_coco', 'arch_data']", default="coco")
    parser.add_argument("--dataset_split", help="Dataset split to process ['train', "\
                        "'eval', 'all']", default="eval")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_split = args.dataset_split
    assert dataset_name in ["coco", "styled_coco", "arch_data"]
    assert dataset_split in ["train", "eval", "all"]

    return dataset_name, dataset_split


def check_existance(database_path):
    """
    Checking if database file already exists. If so, asking the user for confirmation
    before overwritting it
    """

    # if it does not exists, we are good then
    if(not os.path.exists(database_path)):
        return

    # otherwise warning user and expect confirmation
    print(f"Database file {os.path.basename(database_path)} already exists. "\
           "This process will overwrite the previous files.")
    txt = ""
    while(txt not in ['y', 'Y', 'n', 'N']):
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt == "y" or txt == "Y"):
            return
        elif(txt == "n" or txt == "N"):
            print("Exiting...")
            exit()

    return


def preprocess_retrieval_database(dataset_name="arch_data", dataset_split="eval"):
    """
    Preprocessing a certain dataset to be used as database in the pose-based retrieval
    demonstrator. The preprocessing steps include saving bounding box and keypoint
    annotations into compressed .npz files
    """

    # relevant paths and loading sample exp data
    experiments_path = CONFIG["paths"]["experiments_path"]
    database_path = os.path.join(CONFIG["paths"]["database_path"],
                                 f"database_{dataset_name}_gt_{database_split}.pkl")
    exp_path = os.path.join(experiments_path, "test", "experiment_sample")
    exp_data = load_experiment_parameters(exp_path)
    exp_data["dataset"]["dataset_name"] = dataset_name
    exp_data["model"]["model_name"] = "HRNet"
    exp_data["training"]["batch_size"] = 1

    # checking if database file already exists and expecting user confirmation
    check_existance(database_path)

    # loading dataset
    train_loader, valid_loader = load_dataset(exp_data=exp_data, train=True, validation=True,
                                              shuffle_train=False, shuffle_valid=False)

    # loading images, processing vectors and saving
    data = {}
    if(dataset_split == "train"):
        loaders = [train_loader]
    elif(dataset_split == "eval"):
        loaders = [valid_loader]
    elif(dataset_split == "all"):
        loaders = [train_loader, valid_loader]
    for loader in loaders:
        j = len(data.keys())
        for i, (imgs, _, _, metadata) in enumerate(tqdm(loader)):
            joints = metadata["joints"][0]
            center = metadata["center"][0]
            scale = metadata["scale"][0]
            character_name = metadata['character_name'][0]
            img_name = os.path.basename(metadata["image"][0])
            data[f"img_{i+j}"] = {}
            data[f"img_{i+j}"]["img"] = img_name
            data[f"img_{i+j}"]["joints"] = joints
            data[f"img_{i+j}"]["center"] = center
            data[f"img_{i+j}"]["scale"] = scale
            data[f"img_{i+j}"]["character_name"] = character_name

    # saving into a pickle file
    database = {
        "data": data,
        "metadata": {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "created": timestamp()
        }
    }
    with open(database_path, "wb") as file:
        pickle.dump(database, file)

    return


def test_load_pickle(dataset_name="coco"):
    """
    Loading the presaved pickle and displaying the first entries to make sure
    everything was fine
    """

    database_path = os.path.join(CONFIG["paths"]["database_path"],
                                 f"database_{dataset_name}_gt.pkl")
    with open(database_path, "rb") as file:
        database = pickle.load(file)
    data = database["data"]
    metadata = database["metadata"]

    print("\n###############################")
    print(metadata)
    for i, key in enumerate(data.keys()):
        if(i==3):
            break
        print(data[key])
        print(data[key]["joints"].shape)
        print("\n")
    return


if __name__ == "__main__":
    os.system("clear")
    dataset_name, database_split = process_arguments()
    preprocess_retrieval_database(dataset_name, database_split)
    # test_load_pickle(dataset_name)

#
