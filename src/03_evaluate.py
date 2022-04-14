"""
Evaluating a pretrained HRNet model on the complete validation set

@author: Angel Villar-Corrales
"""

import os
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import DataParallel

import data
import data.data_processing as data_processing
import lib.arguments as arguments
import lib.loss as loss
import lib.inference as inference
import lib.model_setup as model_setup
import lib.utils as utils
import lib.metrics as metrics
import lib.pose_parsing as pose_parsing
import lib.visualizations as visualizations
from lib.logger import Logger, log_function, print_
from models.utils.hrnet_config import update_config
from lib.utils import for_all_methods
import CONSTANTS
from CONFIG import CONFIG


@for_all_methods(log_function)
class Evaluator:
    """
    Class for loading an evaluation set, loading a pretrained model
    and evaluating on the validation set

    Args:
    -----
    exp_path: string
        path to the experiment directory
    checkpoint: string
        name of the file containing the desired state dictionaries
    """

    def __init__(self, exp_path, checkpoint=None, dataset_name=None, params=None):
        """ Initializer of the evaluator object """

        # relevant paths
        self.exp_path = exp_path
        self.checkpoint = checkpoint
        self.params = params
        self.plots_path = os.path.join(self.exp_path, "plots")
        self.models_path = os.path.join(self.exp_path, "models")
        self.valid_plots_path = os.path.join(self.plots_path, "valid_plots")
        self.img_path = os.path.join(self.plots_path, f"{dataset_name}_imgs")
        utils.create_directory(self.img_path)

        self.scales = [0.5, 1.0, 1.5, 2.0]
        self.exp_data = utils.load_experiment_parameters(exp_path)
        if(dataset_name is not None):
            self.exp_data["dataset"]["dataset_name"] = dataset_name
        self.dataset_name = self.exp_data["dataset"]["dataset_name"]
        self.kpt_thr = self.exp_data["evaluation"]["in_vis_thr"]

        # macros
        pose_parsing.SKELETON = CONSTANTS.SKELETON_HRNET
        data_processing.TO_COCO_MAP = CONSTANTS.COCO_MAP_HRNET
        data_processing.SKIP_NECK = False
        _ = update_config()
        return

    def setup_model_dataset(self):
        """ Loading the dataset and seting up the pretraiend model """

        # updating alpha and styles if necesary
        if(self.params.alpha is not None):
            print_(f"'Alpha' parameter changing to {self.params.alpha}")
            self.exp_data["dataset"]["alpha"] = self.params.alpha
        if(self.params.styles is not None):
            print_(f"'Styles' parameter changing to {self.params.styles}")
            self.exp_data["dataset"]["styles"] = self.params.styles

        # loading dataset
        self.batch_size = self.exp_data["training"]["batch_size"]
        _, self.valid_loader = data.load_dataset(
                exp_data=self.exp_data,
                train=False,
                validation=True,
                shuffle_train=True,
                shuffle_valid=False
            )
        self.image_size = self.exp_data["dataset"]["image_size"]

        # loading pretrained model
        torch.backends.cudnn.fastest = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # setting up model
        model = model_setup.load_model(self.exp_data, checkpoint=self.checkpoint)
        self.model = DataParallel(model)
        if(self.checkpoint is not None):
            print_(f"Loading checkpoint {self.checkpoint}")
            checkpoint_path = os.path.join(self.exp_path, "models", self.checkpoint)
            self.model = model_setup.load_checkpoint(checkpoint_path, model=self.model,
                                                     only_model=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = self.exp_data["model"]["model_name"]

        # initalizing loss
        self.loss_function = loss.PersonMSELoss()
        return

    @torch.no_grad()
    def evaluate_model(self):
        """ Evaluating the pretrained model on the COCO validation set """
        labels_file = self.valid_loader.dataset.annotations_file
        preds_file = os.path.join(self.exp_path, CONFIG["paths"]["submission"])

        loss_values, accuracies = [], []
        all_boxes, all_preds, image_ids = [], [], []
        utils.reset_predictions_file(self.exp_path)
        self.total_imgs = 0
        for i, (imgs, target, target_weight, metadata) in enumerate(tqdm(self.valid_loader)):

            # setting data to CUDA
            imgs, target = imgs.float(), target.float()
            imgs, target = imgs.to(self.device), target.to(self.device)
            target_weight = target_weight.float().to(self.device)

            # forward pass (normal and flipped), and loss computation
            output = inference.forward_pass(
                    model=self.model,
                    img=imgs,
                    model_name=self.model_name,
                    device=self.device,
                    flip=True
                )
            cur_loss = self.loss_function(output, target, target_weight).item()
            loss_values.append(cur_loss)

            _, avg_acc, cnt, pred = metrics.accuracy(output.cpu().numpy(), target.cpu().numpy())
            accuracies.append(avg_acc)

            # extracting keypoints from network outputs
            centers = metadata["center"].numpy()
            scales = metadata["scale"].numpy()
            score = metadata["score"].numpy()

            output = output.clone().detach()
            keypoints, max_vals, coords = pose_parsing.get_final_preds_hrnet(
                    heatmaps=output.cpu().numpy(),
                    center=centers,
                    scale=scales
                )

            # saving image results if save parameter is activated
            if(self.params.save):
                poses, kpts = pose_parsing.create_pose_from_outputs(
                        dets=output,
                        keypoint_thr=self.kpt_thr
                    )
                for n in range(len(poses)):
                    savepath = os.path.join(
                            self.img_path,
                            f"img_{self.total_imgs}_{metadata['image'][n].split('/')[-1]}"
                        )
                    self.total_imgs = self.total_imgs + 1
                    visualizations.draw_pose(img=imgs[n, :].cpu().numpy(), poses=[poses[n]],
                                             all_keypoints=kpts, preprocess=True,
                                             savepath=savepath, axis_off=True)
                if(self.dataset_name != "arch_data" and i * self.batch_size > 180):
                    break

            # updating submission data
            n_imgs = len(metadata["image"])
            cur_preds = np.zeros((n_imgs, 17, 3), dtype=np.float32)
            cur_boxes = np.zeros((n_imgs, 6))
            cur_preds[:, :, :2] = keypoints[:, :, :2]
            cur_preds[:, :, 2:3] = max_vals
            cur_boxes[:, 0:2] = centers[:, 0:2]
            cur_boxes[:, 2:4] = scales[:, 0:2]
            cur_boxes[:, 4] = np.prod(scales*200, 1)
            cur_boxes[:, 5] = score
            all_preds.append(cur_preds)
            all_boxes.append(cur_boxes)
            # the image id is required by cocoapi
            image_ids += metadata["image_id"].tolist()

            # displaying average loss and precision/recall figures every 1000 images
            if(i == 0):
                continue
            if(i % (2000//self.batch_size) == 0 or i == len(self.valid_loader)-1):
                avg_loss = np.mean(loss_values)
                print_(f"Mean Loss: {avg_loss}")
                print_(f"Accuracy: {np.mean(accuracies)}")
                utils.reset_predictions_file(self.exp_path)
                metrics.generate_submission_hrnet(all_preds, all_boxes, image_ids, preds_file)
                _ = metrics.compute_precision(labels_file=labels_file, preds_file=preds_file)

        # computing precision results on the validation set
        avg_loss = np.mean(loss_values)
        print_(f"Mean Loss: {avg_loss}")
        print_(f"Accuracy: {np.mean(accuracies)}")
        stats = metrics.compute_precision(labels_file=labels_file, preds_file=preds_file)

        self.valid_stats = stats
        utils.save_evaluation_stats(
                stats=self.valid_stats,
                exp_path=self.exp_path,
                detector=False,
                checkpoint=self.checkpoint,
                dataset_name=self.dataset_name,
                alpha=self.exp_data["dataset"]["alpha"],
                styles=self.exp_data["dataset"]["styles"]
            )
        print_(f"Valid Stats: {self.valid_stats}")
        return


if __name__ == "__main__":

    os.system("clear")
    exp_path, checkpoint, dataset, params = arguments.get_directory_argument(
            get_checkpoint=True,
            get_dataset=True
        )

    # initializing logger and logging the beggining of the experiment
    logger = Logger(exp_path)
    message = "Initializing model evaluation."
    logger.log_info(message=message, message_type="new_exp")
    logger.log_info(message=f"  Checkpoint: {checkpoint}", message_type="params")
    logger.log_info(message=f"  Dataset: {dataset}", message_type="params")
    for param, val in vars(params).items():
        logger.log_info(message=f"  {param}: {val}", message_type="params")

    trainer = Evaluator(
            exp_path=exp_path,
            checkpoint=checkpoint,
            dataset_name=dataset,
            params=params
        )
    trainer.setup_model_dataset()
    trainer.evaluate_model()
    logger.log_info(message="Evaluation finished successfully.")


#
