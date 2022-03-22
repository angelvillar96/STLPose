"""
Training (fine-tuning) and Validation of an HRNet Pose Estimation model

EnhancePoseEstimation/src
@author:
"""

import os
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel

import data
import data.data_processing as data_processing
import lib.arguments as arguments
import lib.model_setup as model_setup
import lib.inference as inference
import lib.metrics as metrics
import lib.pose_parsing as pose_parsing
import lib.utils as utils
from lib.logger import Logger, log_function, print_
from lib.utils import for_all_methods
from lib.loss import apply_perceptual_loss, load_perceptual_loss_dict, PersonMSELoss
from models.utils.hrnet_config import update_config
from CONFIG import CONFIG


@for_all_methods(log_function)
class Trainer:
    """
    Class used for training the HRNet model and validating its
    performance for the task of Pose Estimation

    Args:
    -----
    exp_path: string
        path to the experiment directory
    checkpoint: string
        name of the checkpoit to load to resume training
    """

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, params=None):
        """
        Initializer of the trainer object
        """

        # relevant paths
        self.exp_path = exp_path
        self.params = params
        self.models_path = os.path.join(self.exp_path, "models")
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.valid_plots_path = os.path.join(self.plots_path, "valid_plots")
        self.checkpoint = checkpoint
        self.dataset_name = dataset_name
        if(checkpoint is None):
            self.checkpoint_path = None
        else:
            self.checkpoint_path = os.path.join(self.models_path, self.checkpoint)

        self.exp_data = utils.load_experiment_parameters(exp_path)
        cfg = update_config()

        # train/eval parameters
        self.train_loss = 1e18
        self.valid_loss = 1e18
        self.save_frequency = self.exp_data["training"]["save_frequency"]
        self.num_epochs = self.exp_data["training"]["num_epochs"]
        self.scheduler_type = self.exp_data["training"]["scheduler"] \
                              if "scheduler" in self.exp_data["training"] else "plateau"
        self.iterations = 0
        self.cur_epoch = 0

        return


    def load_dataset(self):
        """
        Loading training and validation data-loaders
        """

        # updating alpha, styles and db if necesary
        if(self.params.alpha is not None):
            print_(f"'Alpha' parameter changing to '{self.params.alpha}'")
            self.exp_data["dataset"]["alpha"] = self.params.alpha
        if(self.params.styles is not None):
            print_(f"'Styles' parameter changing to '{self.params.styles}'")
            self.exp_data["dataset"]["styles"] = self.params.styles
        if(self.dataset_name is not None):
            print_(f"'dataset_name' parameter changing to '{self.dataset_name}'")
            self.exp_data["dataset"]["dataset_name"] = self.dataset_name

        self.perceptual_loss_dict = load_perceptual_loss_dict(exp_data=self.exp_data,
                                                              params=self.params)

        self.train_loader,\
         self.valid_loader = data.load_dataset(exp_data=self.exp_data, train=True,
                                               validation=True, shuffle_train=True,
                                               shuffle_valid=False,
                                               percentage=self.params.percentage,
                                               perceptual_loss_dict=self.perceptual_loss_dict)
        self.image_size = self.exp_data["dataset"]["image_size"]

        return


    def setup_model(self):
        """
        Seting up the model, the hardware and model hyperparameters
        """

        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setting up model
        self.model = model_setup.load_model(self.exp_data, checkpoint=self.checkpoint)
        self.model.eval()
        self.model = DataParallel(self.model).to(self.device)
        self.model_name = self.exp_data["model"]["model_name"]

        # setting up model optimizer, learning rate scheduler and loss function
        self.optimizer, self.scheduler = model_setup.setup_optimizer(exp_data=self.exp_data,
                                                                     net=self.model)
        self.loss_function = PersonMSELoss()

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
                                                         only_model=True)
        return


    def training_loop(self):
        """
        Iteratively executing training and validation epochs while saving loss value
        in training logs file
        """

        # initializing training logs is we are starting training
        if(self.checkpoint is None or self.params.resume_training is False):
            self.training_logs = utils.create_train_logs(self.exp_path)
        else:
            self.training_logs = utils.load_train_logs(self.exp_path)

        # iterating for the desired number of epochs
        for epoch in range(self.cur_epoch, self.num_epochs):
            print_(f"########## Epoch {epoch+1}/{self.num_epochs} ##########")
            self.model.eval()
            self.validation_epoch()
            self.model.train()
            self.train_epoch()
            if(self.scheduler_type == "plateau"):
                self.scheduler.step(self.valid_loss)
            elif(self.scheduler_type == "step"):
                self.scheduler.step()
            utils.update_train_logs(self.exp_path, self.training_logs, self.iterations,
                                    train_loss=self.train_loss, valid_loss=self.valid_loss,
                                    train_acc=self.train_acc, valid_acc=self.valid_acc)
            if(epoch % self.save_frequency == 0):
                print_(f"Saving model checkpoint")
                model_setup.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                            scheduler=self.scheduler, epoch=epoch,
                                            exp_path=self.exp_path)

        print_(f"Finished training procedure")
        print_(f"Saving final checkpoint")
        model_setup.save_checkpoint(model=self.model, optimizer=self.optimizer,
                                    scheduler=self.scheduler, epoch=self.num_epochs,
                                    exp_path=self.exp_path, finished=True)

        return


    def train_epoch(self):
        """
        Computing a training epoch: forward and backward pass for the complete training set
        """

        batch_losses, accuracies = [], []

        for i, (imgs, target, target_weight, metadata) in enumerate(tqdm(self.train_loader)):

            # setting input and targets to CUDA
            imgs, target  = imgs.float(), target.float()
            imgs, target = imgs.to(self.device), target.to(self.device)
            target_weight = target_weight.float().to(self.device)

            # forward pass (normal and flipped), and loss computation
            output = inference.forward_pass(model=self.model, img=imgs,
                                            model_name=self.model_name,
                                            device=self.device, flip=False)

            # computing loss as optimization metric
            loss = self.loss_function(output, target, target_weight)
            loss = apply_perceptual_loss(exp_data=self.exp_data, params=self.params,
                                         loss=loss, perceptual_loss=metadata["perceptual_loss"])
            cur_batch_loss = loss.item()
            batch_losses.append(cur_batch_loss)

            # computing accuracy as validation metric
            _, avg_acc, cnt, pred = metrics.accuracy(output.cpu().detach().numpy(),
                                                     target.cpu().numpy())
            accuracies.append(avg_acc)

            # printing small update every 250 mini-batches
            if(i % 250 == 0 and i != 0):
                print_(f"    Batch {i}/{len(self.train_loader)}")
                print_(f"        Loss: {np.mean(batch_losses)}")
                print_(f"        Accuracy: {np.mean(accuracies)}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        self.train_loss = np.mean(batch_losses)
        self.train_acc = np.mean(accuracies)
        print_(f"Train Loss: {self.train_loss}")
        print_(f"Train Accuracy: {self.train_acc}")

        return


    def validation_epoch(self):
        """
        Computing a validation epoch: forward pass for estimating poses and keypoints.
        """

        batch_losses, accuracies = [], []
        all_boxes, all_preds, image_names = [], [], []

        with torch.no_grad():
            for i, (imgs, target, target_weight, metadata) in enumerate(tqdm(self.valid_loader)):

                # we simply use a small subset of 1/5 of total data for validation
                if(i >= len(self.valid_loader) // 5):
                    break

                # setting input and targets to CUDA
                imgs, target  = imgs.float(), target.float()
                imgs, target = imgs.to(self.device), target.to(self.device)
                target_weight = target_weight.float().to(self.device)

                # forward pass (normal and flipped), and loss computation
                output = inference.forward_pass(model=self.model, img=imgs,
                                                model_name=self.model_name,
                                                device=self.device, flip=False)

                # computing loss and accuracy as validation metrics
                loss = self.loss_function(output, target, target_weight)
                loss = apply_perceptual_loss(exp_data=self.exp_data, params=self.params,
                                             loss=loss, perceptual_loss=metadata["perceptual_loss"])
                cur_batch_loss = loss.item()
                batch_losses.append(cur_batch_loss)

                _, avg_acc, cnt, pred = metrics.accuracy(output.cpu().numpy(),
                                                         target.cpu().numpy())
                accuracies.append(avg_acc)

        self.valid_loss = np.mean(batch_losses)
        self.valid_acc = np.mean(accuracies)
        print_(f"Valid Loss: {self.valid_loss}")
        print_(f"Valid Accuracy: {self.valid_acc}")

        return



if __name__ == "__main__":

    os.system("clear")
    exp_path, checkpoint, dataset, \
        params = arguments.get_directory_argument(get_checkpoint=True, get_dataset=True)

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = f"Initializing training procedure"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_params(params=vars(params))

    trainer = Trainer(exp_path=exp_path, checkpoint=checkpoint, dataset_name=dataset, params=params)
    logger.log_params(params=trainer.exp_data)

    trainer.load_dataset()
    trainer.setup_model()
    trainer.training_loop()
    logger.log_info(message=f"Training finished successfully")

#
