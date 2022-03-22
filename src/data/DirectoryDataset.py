"""
Creating a (not-annotated) Dataset by fitting images from a directory

@author: 
"""

import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from CONFIG import CONFIG


class DirectoryDataset:
    """
    Creating a dataset object from images located under the same directory

    Args:
    -----
    datapath: string
        path to the directory where the images are located
    split: string
        Dataset split to consider
    """

    def __init__(self, datapath, split=None, valid_size=0, random_seed=None, shuffle=False):
        """
        Initializer of the dataset object
        """

        assert os.path.exists(datapath)
        assert split in ["train", "validation", None]

        self.datapath = datapath
        self.split = split if split is not None else "train"
        self.valid_size = valid_size
        self.random_seed = CONFIG["random_seed"] if random_seed is None else random_seed

        self.data = os.listdir(self.datapath)
        if(shuffle):
            np.random.shuffle(self.data)
        self.num_imgs = len(self.data)

        return


    def __len__(self):
        """Obtaining the number of images in the dataset"""
        return self.num_imgs


    def __getitem__(self, idx):
        """
        Sampling an image from the database
        """

        image_path = os.path.join(self.datapath, self.data[idx])
        data_numpy = cv2.imread(
            image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        data_numpy = data_numpy.transpose(2,0,1)

        metadata = {
            "image_name": image_path.split("/")[-1],
            "image_path": image_path,
            "image_shape": data_numpy.shape
        }

        return data_numpy, metadata


#
