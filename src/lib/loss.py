
"""
Methods for computing loss functions

EnhancePoseEstimation/src/lib
@author: Angel Villar-Corrales 
"""
import os
import sys
import json
import numpy as np
import os, sys
import json
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import data.data_processing as data_processing
import data.custom_transforms as custom_transforms
import lib.utils as utils
import lib.bounding_box as bbox
import lib.pose_parsing as pose_parsing
from lib.logger import Logger, log_function, print_
from CONFIG import CONFIG

import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class PersonMSELoss(nn.Module):
    """
    Loss function used for training Top-down approaches (loss after human detector)
    """

    def __init__(self, use_target_weight=1):
        """
        Initializer of the loss module
        """

        super(PersonMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

        return


    def forward(self, output, target, target_weight=1):
        """
        Computing the average loss value accross all joints
        """

        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            # loss function without the binary masking
            # loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

            # we avoid computing loss for 'invisible' points by using a binary mask
            loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx, :]),
                    heatmap_gt.mul(target_weight[:, idx, :])
                )

        avg_loss = loss / num_joints

        return avg_loss


def apply_perceptual_loss(exp_data, params, loss, perceptual_loss):
    """
    Adding the corresponding contribution of the perceptual loss if necessary

    Args:
    -----
    exp_data: dictionary
        dict containing the parameters for the current experiment
    params: Namespace
        Parameters from the command line arguments
    loss: torch.Tensor
        loss value computed using input/output of the model
    perceptual_loss: torch.Tensor
        tensor contain the perceptual loss values for the current mini-batch

    Returns
    -------
    total_loss: torch Tensor
        final loss value, including the contribution from the perceptual loss
    """

    # processing whether perceptual loss is considered during training
    use_perceptual_loss = False
    if("perceptual_loss" not in exp_data["training"]):
        exp_data["training"]["perceptual_loss"] = False
    if(params.use_perceptual_loss == True or exp_data["training"]["perceptual_loss"] == True):
        use_perceptual_loss = True
    dataset_name = exp_data["dataset"]["dataset_name"]

    # case for not applying the perceptual_loss
    if(dataset_name != 'styled_coco' or use_perceptual_loss is False):
        return loss

    mean_perc_loss = torch.mean(perceptual_loss).float()

    # case for Lambda weighting in the perceptual loss function
    if(exp_data["training"]["lambda_D"] != None and exp_data["training"]["lambda_P"] != None):
        lambda_D = torch.tensor(exp_data["training"]["lambda_D"])
        lambda_P = torch.tensor(exp_data["training"]["lambda_P"])
        total_loss = loss*lambda_D + mean_perc_loss*lambda_P
        return total_loss

    # case for multiplicative weighting of the perceptual loss
    # obtaining weighting for the perceptual_loss
    if("perceptual_weight" not in exp_data["training"]):
        exp_data["training"]["perceptual_weight"] = "add"
    weighting = exp_data["training"]["perceptual_weight"]
    if(weighting == "add"):
        total_loss = loss + loss * mean_perc_loss
    else:
        print_(f"ERROR! Weighting method '{weighting}' is not supported")
        exit()

    return total_loss


@log_function
def load_perceptual_loss_dict(exp_data, params):
    """
    Loading the dictionary mapping styled image name to perceptual loss

    Args:
    -----
    exp_data: dictionary
        dict containing the parameters for the current experiment
    params: Namespace
        Parameters from the command line arguments

    Returns:
    --------
    perceptual_loss_dict: dictionary
        dict mapping Styled-COCO image name with its corresponding precomputed
        perceptual loss value
    """


    # processing whether perceptual loss is considered during training
    use_perceptual_loss = False
    if("perceptual_loss" not in exp_data["training"]):
        exp_data["training"]["perceptual_loss"] = False
    if(params.use_perceptual_loss == True or exp_data["training"]["perceptual_loss"] == True):
        use_perceptual_loss = True

    # loading dictionary with precomputed perceptual oss values if corresponding
    perceptual_loss_dict = None
    dataset_name = exp_data["dataset"]["dataset_name"]
    alpha = exp_data["dataset"]["alpha"]
    style = exp_data["dataset"]["styles"]
    if(dataset_name == 'styled_coco' and use_perceptual_loss is True):
        print_("Loading dictionary with precomputed perceptual loss values...")
        dict_path = os.path.join(CONFIG["paths"]["dict_path"],
                                 f"perceptual_loss_dict_alpha_{alpha}_styles_{style}.json")
        print_(dict_path)
        if not os.path.exists(dict_path):
            print_("You want to train model with perceptual loss, however the offline losses "\
            f"file '{dict_path}' does not exist.\n"\
            "Please run 'aux_create_offline_perceptual_loss.py' file...")
            sys.exit()
        else:
            perceptual_loss_dict = json.loads(open(dict_path).read())

    return perceptual_loss_dict


#
