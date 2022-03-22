"""
Configuration macros and default argument values

EnhancePseEstimation/src
@author: 
"""

import numpy as np

CONFIG = {
    "paths": {
        "data_path": "/home/corrales/MasterThesis/data",
        "database_path": "/home/corrales/MasterThesis/EnhancedPoseEstimation/databases",
        "experiments_path": "/home/corrales/MasterThesis/EnhancedPoseEstimation/experiments",
        "comparisons_path": "/home/corrales/MasterThesis/EnhancedPoseEstimation/experiments/model_comparison",
        "knn_path": "/home/corrales/MasterThesis/EnhancedPoseEstimation/knn",
        "pretrained_path": "/home/corrales/MasterThesis/EnhancedPoseEstimation/resources",
        "dict_path": "/home/corrales/MasterThesis/data/mapping_dicts",
        "submission": "submission_dict.json"
    },
    "num_workers": 0,
    "random_seed": 13
}

DEFAULT_ARGS = {
    "dataset": {
        "dataset_name": "coco",
        "image_size": 400,
        "alpha": "0.5",
        "styles": "redblack",
        "flip": False,
        "num_joints_half_body": 8,
        "prob_half_body": 0,
        "rot_factor": 0,
        "scale_factor": 0.0,
        "test_set": 'val2017',
        "train_set": 'train2017',
        "shuffle_train": False,
        "shuffle_test": False
    },
    "model": {
        "model_name": "HRNet",
        "detector_name": "faster_rcnn",
        "detector_type": ""
    },
    "training": {
        "num_epochs": 100,
        "learning_rate": 0.001,
        "learning_rate_factor": 0.333,
        "patience": 10,
        "scheduler": "plateau",
        "batch_size": 32,
        "save_frequency": 5,
        "optimizer": "adam",
        "momentum": 0.9,
        "nesterov": False,
        "gamma1": 0.9,
        "gamma2": 0.99,
        "lambda_D": None,
        "lambda_P": None,
        "perceptual_loss": False,
        "perceptual_weight": "add"
    },
    "evaluation": {
      "bbox_thr": 0.5,
      "det_nms_thr": 0.5,
      "img_thr": 0.0,
      "in_vis_thr": 0.2,
      "nms_thr": 1.0,
      "oks_thr": 0.9,
      "use_gt_bbox": True
    }
}

#
