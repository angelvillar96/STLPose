"""
Training (fine-tuning) and Validation a Faster-RCNN.
The Faster-RCNN model provided by PyTorch has been trained on the COCO dataset, but
it does not generalize well to our styled data. We use this file for fine-tuning.
EnhancePoseEstimation/src
@author: 
"""

import os, pdb
import json
import math
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel

from data.data_loaders import get_detection_dataset
import data.data_processing as data_processing
import lib.arguments as arguments
import lib.model_setup as model_setup
import lib.inference as inference
import lib.metrics as metrics
import lib.loss as libloss
import lib.utils as utils
from lib.logger import Logger, log_function, print_
from lib.utils import for_all_methods, load_experiment_parameters
from CONFIG import CONFIG

from lib.detection_coco_utils import get_coco_api_from_dataset
from lib.detection_coco_eval import CocoEvaluator


@for_all_methods(log_function)
class DetectorTrain:
    """
    Class used for training the FasterRCNN model and validating its
    performance for the task of person detection in styled data
    Args:
    -----
    exp_path: string
        path to the experiment directory
    checkpoint: string
        name of the checkpoit to load to resume training
    """

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, params=None):
        """
        Initializer of the training object
        """

        self.exp_path = exp_path
        self.params = params
        self.models_path = os.path.join(self.exp_path, "models", "detector")
        self.exp_data = load_experiment_parameters(exp_path)

        self.checkpoint = checkpoint
        if(checkpoint is None):
            self.checkpoint_path = None
        else:
            self.checkpoint_path = os.path.join(self.models_path, self.checkpoint)

        if(dataset_name is not None):
            self.exp_data["dataset"]["dataset_name"] = dataset_name

        self.pretrained = True
        self.cur_epoch = 0
        self.num_epochs = self.exp_data["training"]["num_epochs"]
        self.save_frequency = self.exp_data["training"]["save_frequency"]
        self.display_frequency = 300
        self.class_ids = [1]
        self.num_classes = len(self.class_ids)

        return


    def load_detection_dataset(self):
        """
        Loading training and validation data-loaders. The data used corresponds to
        images from the (styled) COCO dataset. We only train for certain object classes,
        i.e., we only care about the ids specified by class_ids.
        """

        # updating alpha and styles if necesary
        if(self.params.alpha is not None):
            print_(f"'Alpha' parameter changing to {self.params.alpha}")
            self.exp_data["dataset"]["alpha"] = self.params.alpha
        if(self.params.styles is not None):
            print_(f"'Styles' parameter changing to {self.params.styles}")
            self.exp_data["dataset"]["styles"] = self.params.styles

        self.perceptual_loss_dict = libloss.load_perceptual_loss_dict(exp_data=self.exp_data,
                                                                   params=self.params)

        train_loader,\
        valid_loader  = get_detection_dataset(exp_data=self.exp_data, train=True,
                                              validation=True, shuffle_train=True,
                                              shuffle_valid=False, class_ids=self.class_ids,
                                              perceptual_loss_dict=self.perceptual_loss_dict)

        self.train_loader, self.valid_loader = train_loader, valid_loader

        self.coco = get_coco_api_from_dataset(self.valid_loader.dataset)
        self.iou_types = ["bbox"]

        return


    def load_detector_model(self):
        """
        Loading the pretrained person detector model
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = model_setup.setup_detector(pretrained=self.pretrained,
                                           num_classes=self.num_classes)
        model.eval()
        self.model = DataParallel(model).to(self.device)

        # fitting optimizer and scheduler given exp_data parameters
        optimizer, scheduler = model_setup.setup_optimizer(exp_data=self.exp_data,
                                                           net=self.model)

        lr_factor = self.exp_data["training"]["learning_rate_factor"]
        patience = self.exp_data["training"]["patience"]
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_factor,
        #                                             step_size=patience, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience,
                                                               factor=lr_factor, min_lr=1e-8,
                                                               mode="max", verbose=True)

        self.optimizer, self.scheduler = optimizer, scheduler

        # loading pretraining checkpoint if specified
        if(self.checkpoint is not None):
            print_(f"Loading checkpoint {self.checkpoint}")
            # for resuming a training on the same dataset
            if(self.params.resume_training):
                self.model, self.optimizer, self.scheduler, \
                    self.cur_epoch = model_setup.load_checkpoint(self.checkpoint_path,
                                                                 model=self.model,
                                                                 only_model=False,
                                                                 optimizer=self.optimizer,
                                                                 scheduler=self.scheduler)
                print_(f"Resuming training from epoch {self.cur_epoch}/{self.num_epochs}.")
            # for fine-tuning on a new dataset
            else:
                self.model = model_setup.load_checkpoint(self.checkpoint_path,
                                                         model=self.model,
                                                         only_model=True,
                                                         drop_head=True)
        return


    def training_loop(self):
        """
        Iteratively executing training and validation epochs while saving loss value
        in training logs file
        """

        self.model = self.model.to(self.device)
        if(self.params.resume_training):
            self.training_logs = utils.load_detector_logs(self.exp_path)
        else:
            self.training_logs = utils.create_detector_logs(self.exp_path)

        # iterating for the desired number of epochs
        for epoch in range(self.cur_epoch, self.num_epochs):
            print_(f"########## Epoch {epoch+1}/{self.num_epochs} ##########")
            self.model.eval()
            self.validation_epoch()
            self.model.train()
            self.train_epoch()
            if(self.scheduler is not None):
                self.scheduler.step(self.valid_ap)
                # self.scheduler.step()

            utils.update_detector_logs(self.exp_path, self.training_logs,
                                       train_loss=self.train_loss,
                                       valid_ap=self.valid_ap)

            if(epoch % self.save_frequency == 0):
                print_(f"Saving model checkpoint")
                model_setup.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                            scheduler=self.scheduler, epoch=epoch,
                                            exp_path=self.exp_path, detector=True)

        print_(f"Finished training procedure")
        print_(f"Saving final checkpoint")
        model_setup.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, epoch=self.num_epochs,
                                    exp_path=self.exp_path, finished=True, detector=True)

        return


    def train_epoch(self):
        """
        Computing a training epoch: forward and backward pass for the complete training set
        """

        train_loss = []
        for i, (imgs, metas_) in enumerate(tqdm(self.train_loader)):

            imgs = torch.stack(imgs)
            metas = [{k: v for k, v in t.items()} for t in metas_]
            if self.exp_data['training']['perceptual_loss'] == True:
                perceptual_losses = torch.Tensor([m["perceptual_loss"] for m in metas])

            # preparing annotations for forward pass
            imgs = imgs.to(self.device).float()
            targets = []
            for i, meta in enumerate(metas):
                d = {}
                d["boxes"] = meta["targets"]["boxes"].to(self.device).float()
                d["labels"] = meta["targets"]["labels"].to(self.device).long()
                targets.append(d)

            # forward pass and loss computation
            loss_dict = self.model(imgs / 255, targets)  # important to divide by 255

            loss = sum(loss for loss in loss_dict.values())
            loss = apply_perceptual_loss(exp_data=self.exp_data, params=self.params,
                                         loss=loss, perceptual_loss=perceptual_losses)

            loss_value = loss.item()
            if loss_value == 0 or not math.isfinite(loss_value):
                print_(f"Loss is {loss_value}, skipping image")
                continue

            train_loss.append(loss_value)

            # printing small update every few mini-batches
            if(i % self.display_frequency == 0 and self.display_frequency > 0):
                print_(f"    Batch {i}/{len(self.train_loader)}")
                print_(f"        Loss: {np.mean(train_loss)}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.train_loss = np.mean(train_loss)
        print_(f" Train Loss: {self.train_loss}")

        return


    @torch.no_grad()
    def validation_epoch(self):
        """
        Computing a validation epoch: forward pass for regressing the bbox positions
        and classifying bbox content
        """

        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)

        for i, (imgs, metas_) in enumerate(tqdm(self.valid_loader)):
            imgs = torch.stack(imgs)
            metas = [{k: v for k, v in t.items()} for t in metas_]
            # validation only on 1/5th of the images to avoid overfitting
            if(i >= len(self.valid_loader) // 5):
                break

            # prediciting bounding boxes
            imgs = imgs.to(self.device).float()
            outputs = self.model(imgs / 255)

            # mapping image ids with annotatons and outputs
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            res = {meta["image_id"]: output for meta, output in zip(metas, outputs)}

            # updating evaluation object
            self.coco_evaluator.update(res)

        # accumulate predictions from all images and computing evaluation metric
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.valid_stats = self.coco_evaluator.summarize()["bbox"].tolist()
        self.valid_ap = self.valid_stats[0]

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                       'AR .75', 'AR (M)', 'AR (L)']
        print_("Validation Stats")
        print_(f"{stats_names}")
        print_(f"{self.valid_stats}")

        return



if __name__ == "__main__":

    os.system("clear")
    exp_path, checkpoint, dataset_name,\
         params = arguments.get_directory_argument(get_checkpoint=True,
                                                   get_dataset=True)

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Initializing Faster-RCNN training procedure"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_info(message=f"Checkpoint: {checkpoint}", message_type="params")
    logger.log_info(message=f"Dataset: {dataset_name}", message_type="params")
    for param, val in vars(params).items():
        logger.log_info(message=f"{param}: {val}", message_type="params")

    trainer = DetectorTrain(exp_path=exp_path, checkpoint=checkpoint,
                            dataset_name=dataset_name, params=params)
    logger.log_params(params=trainer.exp_data)


    trainer.load_detection_dataset()
    trainer.load_detector_model()
    trainer.training_loop()

    logger.log_info(message=f"Training of the Faster-RCNN finished successfully")

#
