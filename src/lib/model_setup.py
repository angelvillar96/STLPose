"""
Methods for initializing the model, loading checkpoints, setting up optimizers and loss
functions, and other model-related methods

EnhancePoseEstimation/src/lib
@author: Angel Villar-Corrales 
"""

import os

import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import models
from lib.logger import print_
from lib.utils import create_directory
from CONFIG import CONFIG


def load_model(exp_data, checkpoint=None):
    """
    Creating a new Lightweight OpenPose PE model

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment

    Returns:
    --------
    model: nn.Module
        Lightweight OpenPose model
    """

    model_name = exp_data["model"]["model_name"]

    if(model_name == "OpenPose"):
        num_refinement_stages = exp_data["training"]["num_refinement_stages"]
        model = models.OpenPose(num_refinement_stages=num_refinement_stages,
                                num_channels=128, num_heatmaps=19, num_pafs=38)
        pretrained_path = os.path.join(CONFIG["paths"]["pretrained_path"], "OpenPose",
                                       "checkpoint_iter_370000.pth")
        model.load_pretrained(pretrained_path=pretrained_path)

    elif(model_name == "OpenPoseVGG"):
        model = models.OpenPoseVGG()
        pretrained_path = os.path.join(CONFIG["paths"]["pretrained_path"], "OpenPoseVGG",
                                       "body_pose_model.pth")
        model.load_pretrained(pretrained_path)

    elif(model_name == "HRNet"):
        model = models.PoseHighResolutionNet(is_train=False)
        pretrained_path = os.path.join(CONFIG["paths"]["pretrained_path"], "HRnet",
                                       "pose_hrnet_w32_256x192.pth")
        if(checkpoint is None):
            print_("Loading COCO pretrained weights...")
            model.load_state_dict(torch.load(pretrained_path))

    else:
        raise NotImplementedError("So far only ['OpenPose', 'OpenPoseVGG', 'HRNet']" +\
                                  "models have been implemented")

    return model


def setup_detector(model_name="faster_rcnn", model_type="", pretrained=True,
                   num_classes=1, **kwargs):
    """
    Loading (pretrained) Faster-RCNN with ResNet-50 backbone for person detection

    Args:
    -----
    model_name: string
        name of the detector model to use ['faster_rcnn', 'efficientdet']
    model_type: string
        type of EfficientDet model to load
    pretrained: boolean
        if True, weights pretrained for object detection in COCO are loaded
    num_classes: integer
        number of classes that we want to detect. Model will have (n_classes + 1)
        neurons in the output layer, one for each class and one for 'background'
    """

    assert model_name in ['faster_rcnn', 'efficientdet']
    assert model_type in ["", "d0", "d3"]

    if(model_name == "faster_rcnn"):
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # transfer learning: keeping feature extractor and replacing classifier head
        model = _setup_rcnn_head(model, num_classes=num_classes)

    elif(model_name == "efficientdet"):
        if(model_type is None or model_type=="d0"):
            compound_coef = 0
        elif(model_type=="d3"):
            compound_coef = 3
        anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        anchors_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        model = models.EfficientDet(compound_coef=compound_coef, num_classes=num_classes,
                                    ratios=anchors_ratios, scales=anchors_scales,
                                    threshold=0.5, iou_threshold=0.5)

    else:
        print_(f"Unrecognized person detector name: {model_name}...", message_type="error")
        exit()

    return model


def _setup_rcnn_head(model, num_classes=1, device="cpu"):
    """
    Initializing a new Faster RCNN regression/classification head from scratch
    """

    if(device == "cpu"):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    else:
        in_features = model.module.roi_heads.box_predictor.cls_score.in_features
        model.module.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)


    return model


def setup_optimizer(exp_data, net):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment
    net: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """

    lr = exp_data["training"]["learning_rate"]
    lr_factor = exp_data["training"]["learning_rate_factor"]
    patience = exp_data["training"]["patience"]
    momentum = exp_data["training"]["momentum"]
    optimizer = exp_data["training"]["optimizer"]
    nesterov = exp_data["training"]["nesterov"]
    scheduler = exp_data["training"]["scheduler"] if "scheduler" in exp_data["training"] else "plateau"
    if(optimizer == "adam"):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)

    if(scheduler == "plateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience,
                                                               factor=lr_factor, min_lr=1e-8,
                                                               mode="min", verbose=True)
    elif(scheduler == "step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_factor,
                                                    step_size=patience)
    else:
        scheduler = None

    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, exp_path,
                    finished=False, detector=False):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    detector: boolean
        if True, current checkpoint corresponds to the person detector model. Otherwise
        it corresponds to the pose-estimation model
    """

    if(finished):
        checkpoint_name = f"checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    if(detector):
        savedir = os.path.join(exp_path, "models", "detector")
        if(not os.path.exists(savedir)):
            create_directory(savedir)
        savepath =  os.path.join(savedir, checkpoint_name)
    else:
        savedir = os.path.join(exp_path, "models")
        if(not os.path.exists(savedir)):
            create_directory(savedir)
        savepath = os.path.join(savedir, checkpoint_name)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            }, savepath)

    return


def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """

    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    # loading model parameters. Try catch is used to allow different dicts
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        model.load_state_dict(checkpoint)

    # dropping last layers for Faster RCNN (transfer learning)
    if("drop_head" in kwargs and kwargs["drop_head"] == True):
        print_(f"Re-Initializing Faster R-CNN Head...")
        model = _setup_rcnn_head(model, num_classes=1, device="gpu")

    # returning only the model for transfer learning or returning also optimizer state
    # for continuing training procedure
    if(only_model):
        return model
    optimizer, scheduler = kwargs["optimizer"], kwargs["scheduler"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"]

    return model, optimizer, scheduler, epoch

#
