"""
Creating a dictionary mapping the name of a character to the narrative its
image belongs to

@author: Angel Villar-Corrales 
"""

import os
import json
from tqdm import tqdm

from data.data_loaders import get_detection_dataset
import lib.utils as utils
from CONFIG import CONFIG

MAP = {
    "wm": "Wrestling Mythological",
    "e": "Abduction",
    "bf": "Leading",
    "v": "Pursuit",
    "wa": "Wrestling Agonal"
}

def main():
    """
    Main logic
    """

    # relevant paths
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], "test", "default_coco")
    exp_data = utils.load_experiment_parameters(exp_path)
    dict_path = CONFIG["paths"]["dict_path"]
    utils.create_directory(dict_path)
    savepath = os.path.join(dict_path, "narrative_char_map.json")
    savepath2 = os.path.join(dict_path, "char_narrative_map.json")

    exp_data["dataset"]["dataset_name"] = "arch_data"
    exp_data["training"]["batch_size"] = 1

    train_loader,_  = get_detection_dataset(exp_data=exp_data, train=True,
                                            validation=False, shuffle_train=False,
                                            shuffle_valid=False, valid_size=0)

    mapping_dict = {
        "Wrestling Mythological": [],
        "Abduction": [],
        "Leading": [],
        "Pursuit": [],
        "Wrestling Agonal": []
    }
    mapping_inv = {}

    # iterating dataset processing metadata
    for i, (_, metas) in enumerate(tqdm(train_loader)):
        metas = [{k: v for k, v in t.items()} for t in metas][0]
        img_name = metas["image_name"]
        if("-" in img_name):
            img_narrative_id = img_name.split(".")[0][:-6]
        else:
            img_narrative_id = img_name.split(".")[0][:-4]
        try:
            narrative = MAP[img_narrative_id]
        except Exception as e:
            # print(img_name)
            continue
        char_names = metas["targets"]["arch_labels_str"]

        # updating mapping dicts
        for char in char_names:
            if(char not in mapping_dict[narrative]):
                mapping_dict[narrative].append(char)
            if(char not in mapping_inv.keys()):
                mapping_inv[char] = narrative

    # storing mapping dict
    with open(savepath, "w") as file:
        json.dump(mapping_dict, file)
    with open(savepath2, "w") as file:
        json.dump(mapping_inv, file)
    with open(os.path.join(exp_path, "test_dict.json"), "w") as file:
        json.dump(mapping_dict, file)
    with open(os.path.join(exp_path, "test_inv.json"), "w") as file:
        json.dump(mapping_inv, file)

    return


if __name__ == "__main__":
    os.system("clear")
    main()

#
