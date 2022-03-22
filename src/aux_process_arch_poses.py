"""
Code for processing the ArchData dataset pose annotations from the original CSV
file into a JSON file matching the MS-COCO format

@author: 
"""

import os, json
import argparse
import pdb
from tqdm import tqdm

import numpy as np
import pandas as pd

from CONFIG import CONFIG
import CONSTANTS


def process_arguments():
    """
    Processing command line arguments
    """

    # command line aru
    parser = argparse.ArgumentParser(description='Creating dataset arg parser')
    parser.add_argument('--anns_file', help='Name of the annotations file.' \
        'It must be located under the "data_path/class_arch_poses" directory ',
         default="pose_labels_classarch.csv")
    args = parser.parse_args()
    anns_file_name = args.anns_file

    data_path = CONFIG["paths"]["data_path"]
    anns_file = os.path.join(data_path, "class_arch_poses", anns_file_name)

    if(not os.path.exists(anns_file)):
        print(f"ERROR! Annotation file '{anns_file_name}' not found under the "\
              f"{os.path.join(data_path, 'class_arch_poses')} directory")
        exit()

    return anns_file


def main(anns_fpath):
    """
    Main logic for loadig the CSV file, processing the annotations and saving
    into a JSON with the same format as COCO to later use for training and evaluation

    Args:
    -----
    anns_fpath: string
        path to the csv file containing the provided annotations for keypoints and
        poses in the ArchData dataset
    """

    # loading corresponding data-specific constants
    SKELETON_ARCH_DATA = CONSTANTS.SKELETON_ARCH_DATA
    KPT_NAME_TO_IDX = CONSTANTS.KPT_NAME_TO_IDX_ARCH_DATA
    IDX_TO_KPT_NAME = CONSTANTS.IDX_TO_KPT_NAME_ARCH_DATA#
    REORDER_MAP = CONSTANTS.REORDER_MAP_ARCH_DATA
    ARCHDATA_LBLS_TO_COCO = CONSTANTS.ARCHDATA_LBLS_TO_COCO

    # loading csv file into a pandas dataframe
    if(not os.path.exists(anns_fpath)):
        print(f"ERROR! Annotations file '{anns_fpath}' does not exists")
        exit()
    df = pd.read_csv(anns_fpath)

    # initializing skeleton for the annotations dictionaries
    archdata_keypoints = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    instance_img = {
        "file_name": "",  # name of the image containing the instance
        "full_name": "",  # name of the full image
        "height": -1,     # height of the person instance
        "width": -1,      # width of the person instance
        "id": -1          # id of the instance (links annotations with images)
    }
    instance_anns = {
        "keypoints": [-1]*51,     # list of 51 items: x_coord, y_coord, visibility
        "num_keypoints": -1,      # number of visible keypoints
        "archdata_kpts": [-1]*51, # raw keypoints from csv
        "area": -1,               # area of the bounding box
        "id": -1,                 # unique_id of each person. For ArchData: image_id = id
        "image_id": -1,           # id of the image the annotation belongs to
        "bbox": [-1,-1,-1,-1],     # bbox of the instance. Computed as [xmin, ymin, xmax-xmin, ymax-ymin]
        "character_name": "",     # name of the character
        "iscrowd": 0,             # Instance belongs to crow. Defaults to 0 for ArchData
        "category_id": 1          # corresponding to person
    }
    category= {
      "supercategory": "person",
      "id": 1,
      "name": "person",
      "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"],
      "skeleton": [[16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13], [6,7], [6,8],
                   [7,9], [8,10], [9,11], [2,3], [1,2], [1,3], [2,4], [3,5], [4,6], [5,7]]
    }
    archdata_keypoints["categories"].append(category)


    # iterating each row of the CSV, processing the annotations into our format
    for k, v in tqdm(df.iterrows(), total=len(df)):
        cur_img = {**instance_img}
        cur_anns = {**instance_anns}

        # image metadata
        cur_img["file_name"] = v["body_url"]
        cur_img["full_name"] = v["body_url"]
        cur_img["height"] = v["body_height"]
        cur_img["width"] = v["body_width"]
        cur_img["id"] = int(k)

        # processing raw annotations into vector shape
        width, height = int(v["body_width"]), int(v["body_height"])
        character = v["body_url"].split("_")[0]
        raw_kpts = json.loads(v['keypoints'])
        archdata_kpts = [[0,0,0] for i in range(18)]
        for kpt in raw_kpts:
            kpt_name = kpt["label"]
            pose_vector_id = KPT_NAME_TO_IDX[ARCHDATA_LBLS_TO_COCO[kpt_name]]
            archdata_kpts[pose_vector_id] = [kpt["x"], kpt["y"], 2]

        # converting raw annotations into coco annotations
        coco_kpts = [0] * 17
        num_kpts = 0
        for i, id in enumerate(REORDER_MAP):
            if(id == -1):
                coco_kpts[i] = [0, 0, 0]
            else:
                coco_kpts[i] = archdata_kpts[id]
                if(archdata_kpts[id] != [0,0,0]):
                    num_kpts += 1

        # flattering lists
        archdata_kpts = [kpts[i] for kpts in archdata_kpts for i in range(3)]
        coco_kpts = [kpts[i] for kpts in coco_kpts for i in range(3)]
        # computing bounding box as [xmin, ymin, xmax-xmin, ymax-ymin]
        x = [coco_kpts[3*i] for i in range(17) if(coco_kpts[3*i]>0)]
        y = [coco_kpts[3*i + 1] for i in range(17) if(coco_kpts[3*i+1]>0)]
        x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
        area = (x1-x0)*(y1-y0)
        bbox = [int(x0), int(y0), int(x1-x0), int(y1-y0)]

        # annotations
        cur_anns["keypoints"] = coco_kpts
        cur_anns["num_keypoints"] = num_kpts
        cur_anns["archdata_kpts"] = archdata_kpts
        cur_anns["area"] = int(area)
        cur_anns["id"] = int(k)
        cur_anns["image_id"] = int(k)
        cur_anns["bbox"] = bbox
        cur_anns["character_name"] = character

        # appending current annotations to anns dictionary
        archdata_keypoints["images"].append(cur_img)
        archdata_keypoints["annotations"].append(cur_anns)


    # saving annotations into the json file
    resources_folder = os.path.join(CONFIG["paths"]["data_path"], "annotations_arch_data")
    fname = os.path.join(resources_folder, "arch_data_keypoints.json")
    with open(fname, "w") as f:
        json.dump(archdata_keypoints, f)

    return


if __name__ == "__main__":
    os.system("clear")
    anns_fpath = process_arguments()
    main(anns_fpath=anns_fpath)



#
