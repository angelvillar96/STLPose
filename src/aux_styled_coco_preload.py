"""
Methods for creating metadata to speed up the accessing and loading to the Styled
COCO dataset
"""

import os
from tqdm import tqdm
import json


from data import HRNetCoco
import lib.utils as utils
import lib.arguments as arguments
from CONFIG import CONFIG


def main():
    """
    Main logic
    """

    exp_path, params = arguments.get_directory_argument()
    alpha = params.alpha 
    style = params.styles
    exp_data = utils.load_experiment_parameters(exp_path)

    data_path = CONFIG["paths"]["data_path"]
    labels_path = os.path.join(data_path, "annotations")
    dict_path = CONFIG["paths"]["dict_path"]
    utils.create_directory(dict_path)

    train_dict_path = os.path.join(dict_path, f"train_dict_style_{style}_alpha_{alpha}.json")
    valid_dict_path = os.path.join(dict_path, f"valid_dict_style_{style}_alpha_{alpha}.json")

    if(os.path.exists(train_dict_path) or os.path.exists(valid_dict_path)):
        print(f"Mapping dicts already exists. They will be overwritten " +
                "during the proccess...")
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt != "y" and txt != "Y"):
            return

    # creating mapping dict for the training set
    images_path = os.path.join(data_path, f"images_style_{style}_alpha_{alpha}", "train")
    original_imgs_path = os.path.join(data_path, "original_images", "train2017")
    labels_file = os.path.join(labels_path, "person_keypoints_train.json")
#     dataset = StyledCoco(root=images_path, annFile=labels_file, original_imgs_path=original_imgs_path)
    dataset = HRNetCoco(exp_data=exp_data, root=data_path, img_path=images_path,
                        labels_path=labels_file, is_train=True,
                        is_styled=False)
    train_set_dict = create_mapping_dic(dataset)
    with open(train_dict_path, "w") as file:
        json.dump(train_set_dict, file)


    print("\n\n")

    # creating mapping dict for the validation set set
    images_path = os.path.join(data_path, f"images_style_{style}_alpha_{alpha}", "validation")
    original_imgs_path = os.path.join(data_path, "original_images", "val2017")
    labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
#     dataset = StyledCoco(root=images_path, annFile=labels_file, original_imgs_path=original_imgs_path)
    dataset = HRNetCoco(exp_data=exp_data, root=data_path, img_path=images_path,
                        labels_path=labels_file, is_train=False,
                        is_styled=False)
    valid_set_dict = create_mapping_dic(dataset)
    with open(valid_dict_path, "w") as file:
        json.dump(valid_set_dict, file)

    return


def create_mapping_dic(dataset):
    """
    Creating a dictionary mapping the original image name with the exact name of the
    styled counterpart

    Args:
    -----
    dataset: Dataset
        initialized StyledCoco dataset object

    Returns:
    --------
    mapping_dict: dictionary
        dictionary mapping original image names with the styled image names
    """

    mapping_dict = {}
    coco = dataset.coco
    # styled_image_names = os.listdir(dataset.root)
    styled_image_names = os.listdir(dataset.img_path)

    # iterating all images in the dataset
    for id in tqdm(dataset.image_set_index):
        original_name = coco.loadImgs(id)[0]['file_name'].split('.')[0]
        for cur_styled_img_name in styled_image_names:
            if original_name in cur_styled_img_name:
                new_img_name = cur_styled_img_name
                break
        else:
            print(f"WARNING! Styled image for id: {id} was not found...")
            continue
        mapping_dict[original_name] = new_img_name

    return mapping_dict


if __name__ == "__main__":
    os.system("clear")
    main()
