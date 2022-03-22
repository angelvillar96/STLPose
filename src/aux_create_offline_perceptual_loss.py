"""
Methods for creating offline resources for generating perceptual losses in order
to speed up the accessing and loading to the Styled COCO dataset
"""

import os, pdb
import numpy as np
from tqdm import tqdm
import json
from PIL import Image

import torch
import torchvision

from data import HRNetCoco
import lib.utils as utils
import lib.arguments as arguments
from lib.loss import VGGPerceptualLoss
from lib.logger import Logger, log_function, print_

from CONFIG import CONFIG

@log_function
def main():
    """
    Main logics
    """

    # processing command line arguments and intializing logger
    exp_path, params = arguments.get_directory_argument()
    logger = Logger(exp_path)
    message = f"Initializing training procedure"
    logger.log_info(message=message, message_type="new_exp")
    logger.log_params(params=vars(params))


    # relevant paths and parameters
    exp_data = utils.load_experiment_parameters(exp_path)
    dataset_name = exp_data["dataset"]["dataset_name"]
    alpha = exp_data["dataset"]["alpha"]
    style = exp_data["dataset"]["styles"]
    data_path = CONFIG["paths"]["data_path"]
    dict_path = CONFIG["paths"]["dict_path"]
    fpath = os.path.join(dict_path, f"perceptual_loss_dict_alpha_{alpha}_styles_{style}.json")

    # making sure parameters are correct

    if(dataset_name != "styled_coco"):
        print_("Perceptual loss can only be computed for 'Styled-COCO'...")
        exit()
    if(os.path.exists(fpath)):
        print(f"Perceptual Loss dict already exists. They will be overwritten " +
                "during the proccess...")
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt != "y" and txt != "Y"):
            return


    print_(f"Processing: dataset {dataset_name} with alpha {alpha} and style {style}...")

    ## creating the datapaths for train
    img_path = os.path.join(data_path, f"images_style_{style}_alpha_{alpha}", "train")
    original_images_path = os.path.join(data_path, "coco", "original_images", "train2017")
    train_perceptual_loss_dict = create_perceptual_loss_dict(img_path, original_images_path)

    print("\n\n")

    ## creating the datapaths for train
    img_path = os.path.join(data_path, f"images_style_{style}_alpha_{alpha}", "validation")
    original_images_path = os.path.join(data_path, "coco", "original_images", "val2017")
    valid_perceptual_loss_dict = create_perceptual_loss_dict(img_path, original_images_path)

    train_perceptual_loss_dict.update(valid_perceptual_loss_dict)

    with open(fpath, 'w') as f:
        json.dump(train_perceptual_loss_dict, f)

    return

@log_function
def create_perceptual_loss_dict(images_path, original_images_path):
    """
    Creating a dictionary for the perceptual loss between the original image
    with the it's styled counterpart

    Args:
    -----
    images_path: str
        path to the StyledCoco dataset
    original_images_path: str
        path to the OriginalCoco dataset

    Returns:
    --------
    perceptual_loss_dict: dictionary
        dictionary mapping the perceptual loss of original image
        with it's styled counterpart image
    """

    original_list = os.listdir(original_images_path)
    original_list = [i for i in original_list if '.ipynb_checkpoints' != i]
    styled_list = os.listdir(images_path)
    styled_list = [i for i in styled_list if '.ipynb_checkpoints' != i]

    ## creating the perceptual loss class member
    vgg_perceptual_loss = VGGPerceptualLoss()

    count = 0
    perceptual_loss_dict = {}
    for o_name in tqdm(original_list):
        count += 1
        if count % 1000 == 0:
            print ('{} images done'.format(str(count)))

        for i in styled_list:
            if o_name.split('.')[0] in i:
                s_name = i

        o_path = os.path.join(original_images_path, o_name)
        s_path = os.path.join(images_path, s_name)

        o_image = np.array(Image.open(o_path).convert('RGB'))
        s_image = np.array(Image.open(s_path).convert('RGB'))

        #show_images_in_parallel(o_image, s_image)

        o_array = o_image.transpose(2,0,1)/255.0
        s_array = s_image.transpose(2,0,1)/255.0

        o_tensor = torch.Tensor(o_array).unsqueeze(0)
        s_tensor = torch.Tensor(s_array).unsqueeze(0)

        vgg_loss_value = vgg_perceptual_loss.forward(o_tensor, s_tensor)
        perceptual_loss_dict[s_name] = vgg_loss_value.item()
        
#         if count >=5:
#             break

    return perceptual_loss_dict

if __name__ == "__main__":
    os.system("clear")
    main()

#
