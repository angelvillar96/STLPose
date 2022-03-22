#!/usr/bin/env python
# coding: utf-8


import os, pdb
import sys
import json
import math
import time
from tqdm import tqdm
from argparse import Namespace
sys.path.append("..")

import numpy as np
import torch
from torch.nn import DataParallel

from data.data_loaders import get_detection_dataset
import data.data_processing as data_processing
import lib.arguments as arguments
import lib.model_setup as model_setup
import lib.inference as inference
import lib.metrics as metrics
import lib.loss as loss, apply_perceptual_loss
from lib.logger import print_
import lib.utils as utils
from lib.utils import for_all_methods, load_experiment_parameters
from CONFIG import CONFIG

from lib.detection_coco_utils import get_coco_api_from_dataset
from lib.detection_coco_eval import CocoEvaluator


# In[5]:


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

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, Lambda_D=None,
                 Lambda_P=None, params=None):
        """
        Initializer of the training object
        """

        self.exp_path = exp_path
        self.Lambda_D = Lambda_D
        self.Lambda_P = Lambda_P
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

        self.perceptual_loss_dict = loss.load_perceptual_loss_dict(exp_data=self.exp_data,
                                                                   params=self.params)

        self.pretrained = True
        self.cur_epoch = 0
        self.num_epochs = 5
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

        train_loader,        valid_loader  = get_detection_dataset(exp_data=self.exp_data, train=True,
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

        # iterating for the desired number of epochs
        for epoch in range(self.cur_epoch, self.num_epochs):
            print_(f"########## Epoch {epoch+1}/{self.num_epochs} ##########")
            self.model.eval()
            self.validation_epoch()
            self.model.train()
            self.train_epoch()
            if(self.scheduler is not None):
                self.scheduler.step(self.valid_ap)

        print_(f"Finished training procedure")

        return self.valid_ap


    def train_epoch(self):
        """
        Computing a training epoch: forward and backward pass for the complete training set
        """

        train_loss = []
        for i_batch, (imgs, metas_) in enumerate(tqdm(self.train_loader)):

            imgs = torch.stack(imgs)
            metas = [{k: v for k, v in t.items()} for t in metas_]

            # preparing annotations for forward pass
            imgs = imgs.to(self.device).float()

            batch_perceptual_loss = []
            targets = []
            for i, meta in enumerate(metas):
                d = {}
                d["boxes"] = meta["targets"]["boxes"].to(self.device).float()
                d["labels"] = meta["targets"]["labels"].to(self.device).long()
                targets.append(d)
                batch_perceptual_loss.append(float(meta["perceptual_loss"]))

            # Average batch_perceptual_loss
            avg_batch_perceptual_loss = np.average(np.array(batch_perceptual_loss))

            # forward pass and loss computation
            loss_dict = self.model(imgs / 255, targets)  # important to divide by 255

            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()
            if loss_value == 0 or not math.isfinite(loss_value):
                print_(f"Loss is {loss_value}, skipping image")
                continue

            ### Adding the perceptual loss here?
            if(self.exp_data["dataset"]["dataset_name"]) == 'styled_coco' and self.params.use_perceptual_loss:
                loss_value = self.Lambda_D * loss_value + self.Lambda_P * avg_batch_perceptual_loss

            train_loss.append(loss_value)

            # printing small update every few mini-batches
            if(i % self.display_frequency == 0 and self.display_frequency > 0):
                print_(f"    Batch {i}/{len(self.train_loader)}")
                print_(f"        Loss: {np.mean(train_loss)}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i_batch >= 25:
                print ("Breaking here for mini-subset training")
                break

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


# In[18]:


import optuna
def objective(trial):
    print("Optimize Start")

    #number of the unit
    Lambda_D = trial.suggest_uniform("Lambda_D", 0, 1)
    Lambda_P = trial.suggest_uniform("Lambda_P", 0, 1)

    exp_path = "styled_coco_redblackcombined_alpha_05/experiment_2020-11-02_21-59-12"
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], exp_path)
    checkpoint = None
    dataset_name = "styled_coco"
    params = {
        "save": False,
        "resume_training": False,
        "use_perceptual_loss": True
    }
    params = Namespace(**params)

    trainer = DetectorTrain(exp_path=exp_path, checkpoint=checkpoint,
                            dataset_name=dataset_name, Lambda_D=Lambda_D,
                            Lambda_P=Lambda_P, params=params)


    trainer.load_detection_dataset()
    trainer.load_detector_model()
    valid_ap = trainer.training_loop()

    return valid_ap


# In[ ]:


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)


# In[27]:


import time
import pickle
timestr = time.strftime("%Y%m%d-%H%M%S")
study_name = 'object_detection_' + timestr + '.pkl'
study_path = os.path.join('/localhome/ronak/efi/code/EnhancePoseEstimation/lambda_study', study_name)
with open(study_path, 'wb') as out:
    pickle.dump(study, out)


#
