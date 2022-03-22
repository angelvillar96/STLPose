"""
Qualitatively testing pose-based image retrieval.
Samples a few poses and retrievas the most similar ones, saving results in a directory

@
"""

import os, shutil
import argparse
import pickle
from tqdm import tqdm

import cv2
import numpy as np

from lib.arguments import process_retrieval_arguments
from lib.logger import Logger, log_function, print_
from lib.pose_database import load_database, load_knn, process_pose_vector, \
                              get_neighbors_idxs
import lib.pose_parsing as pose_parsing
import lib.transforms as custom_transforms
from lib.utils import create_directory, for_all_methods, load_experiment_parameters,\
                      load_character_narrative_maps
from lib.visualizations import visualize_image, draw_pose
import CONSTANTS
from CONFIG import CONFIG


@log_function
def retrieval_experiment(exp_directory, database_file, approach="all_kpts",
                         normalize=True, num_retrievals=10, num_exps=5,
                         retrieval_method="knn", penalization="none", shuffle=False,
                         **kwargs):
    """
    Main orquestrator for a retrieval experiment. Some random

    Args:
    -----
    exp_directory: string
        path to the root of the experiment directory
    dataset_name: list
        list with the names of the databases used for retrieval
    approach: string
        Approach (keypoints) used to measure similarity
    normalize: boolean
        If True, normalized pose vectors are used
    num_retrievals: integer
        number of elements to fetch from the database
    num_exps: integer
        number of query images to use => number of times to repeat the experiment
    retrieval_method: string
        method used to compuite the similarity
    penalization: string
        strategy followed to penalize points that are not present in query or dataset
    shuffle: boolean
        If true, query skeletons are sampled at random from the database
    """

    # relevant paths and macros
    pose_coco = CONSTANTS.SKELETON_HRNET
    pose_arch = CONSTANTS.SKELETON_SIMPLE
    pose_parsing.SKELETON = pose_arch
    plots_path = os.path.join(exp_directory, "plots",
                              f"retrieval_exps_{retrieval_method}_{approach}_{penalization}")
    if(os.path.exists(plots_path)):
        print(f"Retrieval Image-examples already exist.\n"\
              "This process will overwrite previous results.")
        txt = input("Do you want to proceed? (y/n)\n")
        if(txt != "y" and txt != "Y"):
            return
        shutil.rmtree(plots_path)
    create_directory(plots_path)

    data_path = os.path.join(CONFIG["paths"]["data_path"], "class_arch_poses", "characters")

    # loading database and retrieval resources
    knn, database, features = load_knn(database_file=database_file)
    keys_list = list(database.keys())
    n_entries = len(keys_list)

    char_to_narr, _ = load_character_narrative_maps()

    # sampling query indicies and computing retrievals
    if(shuffle):
        indices = np.random.randint(low=0, high=n_entries, size=num_exps)
    else:
        indices = np.arange(num_exps)

    for i, ind in enumerate(tqdm(indices)):
        # fetching query data
        query = database[keys_list[ind]]
        query_img = query["img"]
        img_path = os.path.join(data_path, query_img)
        query_joints = query["joints"].numpy()
        cur_character = query["character_name"]
        cur_narrative = char_to_narr[cur_character]
        pose_vector = process_pose_vector(vector=query_joints, approach=approach,
                                          normalize=normalize)#[np.newaxis,:]

        # obtaining similar poses using knn
        idx, dists = get_neighbors_idxs(pose_vector, k=num_retrievals, approach=approach,
                                        retrieval_method=retrieval_method,
                                        penalization=penalization, knn=knn,
                                        database=features)
        retrievals = [database[keys_list[j]] for j in idx]

        # saving query and results
        img_name = f"exp_{i+1}_query_{cur_character}_{cur_narrative}.png"
        savepath1 = os.path.join(plots_path, img_name)
        savepath2 = os.path.join(plots_path, f"skel_{img_name}")
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        pose = custom_transforms.transform_preds(
            query["joints"], query["center"][0].numpy(), query["scale"][0].numpy(), [192, 256]
        )
        pose[:, -1] = query["joints"][:, -1]
        pose = np.array([pose[:,1], pose[:,0], pose[:,2]]).T
        visualize_image(img=data_numpy, savepath=savepath1, axis_off=True)
        draw_pose(img=data_numpy, poses=[np.arange(19)], all_keypoints=pose,
                  savepath=savepath2, axis_off=True)
        # draw_skeleton(kpts=query_joints, savepath=savepath2, title=title)

        for j,ret in enumerate(retrievals):
            # data and metadata
            cur_character = ret["character_name"]
            cur_narrative = char_to_narr[cur_character]
            img_name = f"exp_{i+1}_retrieval_{j+1}_{cur_character}_{cur_narrative}.png"
            savepath1 = os.path.join(plots_path, img_name)
            savepath2 = os.path.join(plots_path, f"skel_{img_name}")
            image_path = os.path.join(data_path, ret["img"])
            data_numpy = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)

            # preprocessing keypoints
            pose = custom_transforms.transform_preds(
                ret["joints"], ret["center"][0].numpy(), ret["scale"][0].numpy(), [192, 256]
            )
            pose[:, -1] = ret["joints"][:, -1]
            pose = np.array([pose[:,1], pose[:,0], pose[:,2]]).T
            visualize_image(img=data_numpy, savepath=savepath1, axis_off=True)
            draw_pose(img=data_numpy, poses=[np.arange(19)], all_keypoints=pose,
                      savepath=savepath2, axis_off=True)
            # draw_skeleton(kpts=ret["joints"].numpy(), savepath=savepath, title=title)
    return




if __name__ == "__main__":
    os.system("clear")

    # processing command line arguments
    args = process_retrieval_arguments()

    # initializing logger
    logger = Logger(args.exp_directory)
    message = f"Initializing retrieval tests..."
    logger.log_info(message=message, message_type="new_exp")
    logger.log_info(message="Parameters:")
    logger.log_info(message="-----------")
    params = vars(args)
    for p in params:
        message = f"    - {p}: {params[p]}"
        logger.log_info(message=message)

    #  starting retrieval experiment
    retrieval_experiment(**vars(args))


    #
