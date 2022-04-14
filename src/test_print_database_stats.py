"""
Loading the dataset and displaying the stats to then include in the report

@author: Angel Villar-Corrales

TODO: Rework this file. e.g. dataset is hardwired
"""

import os

import numpy as np

from data.data_loaders import get_detection_dataset, load_dataset
from lib.utils import load_experiment_parameters, create_directory
from lib.visualizations import visualize_bbox
from CONFIG import CONFIG



def get_database_stats():
    """
    Loading database, fetching stats and printing
    """

    # experingmet path and data
    # DB_NAME = "coco"
    # DB_NAME = "styled_coco"
    DB_NAME = "arch_data"
    # DB_NAME = "combined"
    EXP_PATH = os.path.join(CONFIG["paths"]["experiments_path"],
                           "detector_tests",
                           "hrnet_notebook")
    exp_data = load_experiment_parameters(EXP_PATH)
    exp_data["dataset"]["dataset_name"] = DB_NAME
    plots_path = os.path.join(EXP_PATH, "plots", "report")
    create_directory(plots_path)


    # loading detection DB
    train_loader, \
    valid_loader  = get_detection_dataset(exp_data=exp_data, train=True,
                                          validation=True, shuffle_train=True,
                                          shuffle_valid=True, class_ids=[1])
    train_db, valid_db = train_loader.dataset, valid_loader.dataset
    n_imgs_train, n_imgs_valid = len(train_db), len(valid_db)

    n_train_persons = np.sum([cur_data['targets']['boxes'].shape[0] for cur_data in train_db.data])
    n_valid_persons = np.sum([cur_data['targets']['boxes'].shape[0] for cur_data in valid_db.data])

    # loading pose estimation DB
    train_loader, valid_loader = load_dataset(exp_data=exp_data, train=True,
                                              validation=True, shuffle_train=True,
                                              shuffle_valid=False)
    train_db, valid_db = train_loader.dataset, valid_loader.dataset
    n_instances_train, n_instances_valid = len(train_db), len(valid_db)


    # printing stats
    print(f"Images:")
    print(f"    Train: {n_imgs_train}")
    print(f"    Train Persons: {n_train_persons}")
    print(f"    Valid: {n_imgs_valid}")
    print(f"    Valid Persons: {n_valid_persons}")
    print(f"    Total: {n_imgs_train + n_imgs_valid}")
    print(f"    Total Persons: {n_train_persons + n_valid_persons}")
    print(f"Instances:")
    print(f"    Train: {n_instances_train}")
    print(f"    Valid: {n_instances_valid}")
    print(f"    Total: {n_instances_train + n_instances_valid}")
    return
    # fetching few images and saving them for the report
    for i, (imgs, metas) in enumerate(valid_loader):
        if(i > 50):
            break
        arch_label_ids = metas["targets"]["arch_labels"]
        arch_labels = [train_db.arch_labels_map[int(id)] for id in arch_label_ids]
        img_disp = imgs[0,:].numpy().transpose(1,2,0)
        visualize_bbox(img=img_disp, boxes=metas["targets"]["boxes"][0],
                       labels=metas["targets"]["labels"][0], axis_off=True,
                       savepath=os.path.join(plots_path, f"img_{i+1}.png"),
                       arch_labels=arch_labels)

    return


if __name__ == "__main__":
    os.system("clear")
    get_database_stats()

#
