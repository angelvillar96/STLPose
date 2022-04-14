"""
Training (fine-tuning) and Validation of an HRNet Pose Estimation model

@author: Angel Villar-Corrales
"""

import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

import data
import lib.arguments as arguments
import lib.model_setup as model_setup
import lib.inference as inference
import lib.metrics as metrics
import lib.utils as utils
from lib.logger import Logger, log_function, print_
from lib.utils import for_all_methods
from lib.loss import apply_perceptual_loss, load_perceptual_loss_dict, PersonMSELoss
from models.utils.hrnet_config import update_config


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
        """ Initializer of the trainer object """
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
        _ = update_config()

        tboard_logs = os.path.join(self.exp_path, "tboard_logs")
        self.writer = SummaryWriter(tboard_logs)

        # train/eval parameters
        self.train_loss = 1e18
        self.valid_loss = 1e18
        self.save_frequency = self.exp_data["training"]["save_frequency"]
        self.num_epochs = self.exp_data["training"]["num_epochs"]
        self.scheduler_type = self.exp_data["training"].get("scheduler", "plateau")
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

        self.perceptual_loss_dict = load_perceptual_loss_dict(exp_data=self.exp_data, params=self.params)
        self.train_loader, self.valid_loader = data.load_dataset(
                exp_data=self.exp_data,
                train=True,
                validation=True,
                shuffle_train=True,
                shuffle_valid=False,
                percentage=self.params.percentage,
                perceptual_loss_dict=self.perceptual_loss_dict
            )
        self.image_size = self.exp_data["dataset"]["image_size"]
        return

    def setup_model(self):
        """ Seting up the model, the hardware and model hyperparameters """
        # setting up the device
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setting up model
        self.model = model_setup.load_model(self.exp_data, checkpoint=self.checkpoint)
        self.model.eval()
        self.model = DataParallel(self.model).to(self.device)
        self.model_name = self.exp_data["model"]["model_name"]

        # setting up model optimizer, learning rate scheduler and loss function
        self.optimizer, self.scheduler = model_setup.setup_optimizer(exp_data=self.exp_data, net=self.model)
        self.loss_function = PersonMSELoss()

        # loading pretraining checkpoint if specified
        if(self.checkpoint is not None):
            print_(f"Loading checkpoint {self.checkpoint}")
            # for resuming a training on the same dataset
            if(self.params.resume_training):
                self.model, self.optimizer, self.scheduler, self.cur_epoch = model_setup.load_checkpoint(
                        self.checkpoint_path,
                        model=self.model,
                        only_model=False,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler
                    )
                print_(f"Resuming training from epoch {self.cur_epoch}/{self.num_epochs}.")
            # for fine-tuning on a new dataset
            else:
                self.model = model_setup.load_checkpoint(
                        self.checkpoint_path,
                        model=self.model,
                        only_model=True
                    )
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
            self.model.eval()
            self.validation_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)
            if(self.scheduler_type == "plateau"):
                self.scheduler.step(self.valid_loss)
            elif(self.scheduler_type == "step"):
                self.scheduler.step()
            # updating training logs dictionary
            utils.update_train_logs(self.exp_path, self.training_logs, self.iterations,
                                    train_loss=self.train_loss, valid_loss=self.valid_loss,
                                    train_acc=self.train_acc, valid_acc=self.valid_acc)
            # saving model checkpoint
            if(epoch % self.save_frequency == 0):
                print_("Saving model checkpoint")
                model_setup.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        exp_path=self.exp_path
                    )
            # updating Tensorboard
            self.writer.add_scalars('pose_results/COMB_loss', {
                'train_loss': self.train_loss,
                'eval_loss': self.valid_loss,
            }, epoch+1)
            self.writer.add_scalars('pose_results/COMB_acc', {
                'train_acc': self.train_acc,
                'eval_acc': self.valid_acc,
            }, epoch+1)

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        model_setup.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.num_epochs,
                exp_path=self.exp_path,
                finished=True
            )
        return

    def train_epoch(self, epoch):
        """ Computing a training epoch: forward and backward pass for the complete training set """

        batch_losses, accuracies = [], []
        progress_bar = tqdm(enumerate(self.train_loader))
        for i, (imgs, target, target_weight, metadata) in progress_bar:

            # setting input and targets to CUDA
            imgs, target = imgs.float(), target.float()
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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # computing metrics
            cur_batch_loss = loss.item()
            batch_losses.append(cur_batch_loss)
            _, avg_acc, cnt, pred = metrics.accuracy(output.cpu().detach().numpy(), target.cpu().numpy())
            accuracies.append(avg_acc)

            # logging
            mean_loss = np.mean(batch_losses)
            mean_acc = np.mean(accuracies)
            progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: Mean Loss {mean_loss}. Mean Acc {mean_acc} ")
            iter_ = len(self.train_loader) * epoch + i
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                self.writer.add_scalar('pose_train/loss', mean_loss, global_step=iter_)
                self.writer.add_scalar('pose_train/acc', mean_acc, global_step=iter_)

        self.train_loss = np.mean(batch_losses)
        self.train_acc = np.mean(accuracies)
        print_(f"Train Loss: {self.train_loss}")
        print_(f"Train Accuracy: {self.train_acc}")
        return

    @torch.no_grad()
    def validation_epoch(self, epoch):
        """ Computing a validation epoch: forward pass for estimating poses and keypoints """

        batch_losses, accuracies = [], []
        progress_bar = tqdm(enumerate(self.valid_loader))
        for i, (imgs, target, target_weight, metadata) in progress_bar:

            # we simply use a small subset of 1/5 of total data for validation
            if(i >= len(self.valid_loader) // 5):
                break

            # setting input and targets to CUDA
            imgs, target = imgs.float(), target.float()
            imgs, target = imgs.to(self.device), target.to(self.device)
            target_weight = target_weight.float().to(self.device)

            # forward pass (normal and flipped), and loss computation
            output = inference.forward_pass(
                    model=self.model,
                    img=imgs,
                    model_name=self.model_name,
                    device=self.device,
                    flip=False
                )

            # computing loss and accuracy as validation metrics
            loss = self.loss_function(output, target, target_weight)
            loss = apply_perceptual_loss(
                    exp_data=self.exp_data,
                    params=self.params,
                    loss=loss,
                    perceptual_loss=metadata["perceptual_loss"]
                )
            cur_batch_loss = loss.item()
            batch_losses.append(cur_batch_loss)
            _, avg_acc, cnt, pred = metrics.accuracy(output.cpu().numpy(), target.cpu().numpy())
            accuracies.append(avg_acc)

            # logging
            mean_loss = np.mean(batch_losses)
            mean_acc = np.mean(accuracies)
            progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: Mean Loss {mean_loss}. Mean Acc {mean_acc}")
            iter_ = (len(self.valid_loader) // 5) * epoch + i
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                self.writer.add_scalar('pose_valid/loss', mean_loss, global_step=iter_)
                self.writer.add_scalar('pose_valid/acc', mean_acc, global_step=iter_)

        self.valid_loss = np.mean(batch_losses)
        self.valid_acc = np.mean(accuracies)
        print_(f"Valid Loss: {self.valid_loss}")
        print_(f"Valid Accuracy: {self.valid_acc}")
        return


if __name__ == "__main__":

    os.system("clear")
    exp_path, checkpoint, dataset, params = arguments.get_directory_argument(
            get_checkpoint=True,
            get_dataset=True
        )

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = "Initializing training procedure"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_params(params=vars(params))

    trainer = Trainer(exp_path=exp_path, checkpoint=checkpoint, dataset_name=dataset, params=params)
    logger.log_params(params=trainer.exp_data)

    trainer.load_dataset()
    trainer.setup_model()
    trainer.training_loop()
    logger.log_info(message="Training finished successfully")

#
