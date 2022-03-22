"""
Loading data and model to compute a cyclic learning rate test.
This auxiliar script should be called prior to training so as to find an
adecuate learning rate for the training procedure.

@author: 
"""

import os, pdb
import math
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel
from torch_lr_finder import LRFinder

import lib.arguments as arguments
from lib.visualizations import visualize_bbox
from lib.logger import Logger, log_function, print_
from lib.utils import for_all_methods, load_experiment_parameters
from CONFIG import CONFIG

DetectorTrainer = __import__('02_train_faster_rcnn')


@for_all_methods(log_function)
def find_learning_rate(exp_data, checkpoint):
    """
    Method for finding the best learning rate to train a model with certain specs
    """

    # using 02_train_faster_rcnn interface to load model and data_loader
    trainer = DetectorTrainer(exp_path=exp_path, checkpoint=checkpoint)
    trainer.load_detection_dataset()
    trainer.load_detector_model()

    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1e-7,
                                momentum=0.9, weight_decay=0.0005)

    # using learning rate finder
    lr_finder = LRFinder(trainer.model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainer.valid_loader, end_lr=1, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph

    return



if __name__ == "__main__":
    os.system("clear")
    exp_path, checkpoint = arguments.get_directory_argument(get_checkpoint=True)

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Initializing procedure for finding optimum learning rate"
    logger.log_info(message=message, message_type="new_exp")

    find_learing_rate(exp_path=exp_path, checkpoint=checkpoint)

    logger.log_info(message=f"Learning rate finding procedure finished successfully")
