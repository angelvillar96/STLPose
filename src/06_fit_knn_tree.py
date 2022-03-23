"""
Fitting a k-nearest neighbor tree for pose similarity and retrieval purposes

@author: Angel Villar-Corrales 
"""

import os
import argparse
import pickle

import numpy as np
import hnswlib

from CONFIG import CONFIG


def process_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", help="Name of the file contaning the "\
                        "preprocessed databaser", required=True)
    parser.add_argument("--metric", help="Metric used for retrieval: ['euclidean_distance',"\
                        " 'cosine_similarity'].", default="euclidean_distance")
    parser.add_argument("--approach", help="Approach used to consider similarity.\n"\
                        "    'upper_body': only considering upper-body keypoints.\n"\
                        "    'full_body': considering lower and upper body (not head).\n"\
                        "    'all_kpts': considering all keypoints.", default="full_body")
    parser.add_argument("--normalize", help="If True, pose vectors are L-2 normalized.",
                        default="True")
    args = parser.parse_args()

    args.database_file = os.path.join(CONFIG["paths"]["database_path"], args.database_file)
    args.normalize = (args.normalize == "True")
    metric = args.metric
    approach = args.approach

    # ensuring correct values
    assert os.path.exists(args.database_file)
    assert metric in ["euclidean_distance", "cosine_similarity"]#
    assert approach in ["upper_body", "full_body", "all_kpts"]

    return args


def load_data(database_file):
    """
    Loading pickled objects with the preprocess data

    Args:
    -----
    database_file: string
        path to the pickle file containing the preprocessed pose database

    Returns:
    --------
    data: dictionary
        dictionary containing data and metadata from all desired datasets
    """

    all_dicts = []
    data = {}

    # loading data from all datasets
    with open(database_file, "rb") as file:
        database = pickle.load(file)
    if("data" in database.keys()):
        cur_data = database["data"]
    else:
        cur_data = database
    all_dicts.append(cur_data) # merging dictionaries

    # merging all dictionaries
    offset = 0
    for cur_dict in all_dicts:
        n_imgs = len(cur_dict.keys())
        cur_data = {f"img_{i+offset}":cur_dict[f"img_{i}"] for i in range(n_imgs)}
        data = {**data, **cur_data}
        offset = offset + n_imgs

    return data


def process_data(data, approach, normalize):
    """
    Processing the data, keeping only desired keypoints

    Args:
    -----
    data: dictionary
        dictionary containing data and metadata from all desired datasets
    approach: string
        approach for measuring the similarity
    normalize: boolean
        If True, pose vectors are L-2 normalized

    Returns:
    --------
    processed_features: numpy array
        Array containing the data, already preprocessed and ready to be fit to the knn
    """

    # obtaining just kpt data
    joint_data = [data[k]["joints"].numpy() for k in data.keys()]
    joint_data = np.array(joint_data)

    # obtaining inidices for desired keypoints
    if(approach == "all_kpts"):
        kpt_idx = np.arange(17)  # all keypoints
    elif(approach == "full_body"):
        kpt_idx = np.arange(5, 17)  # from shoulders to ankles
        kpt_idx = np.append(kpt_idx, 0)  # adding nose
    elif(approach == "upper_body"):
        kpt_idx = np.arange(5, 13)  # from shoulders to  hips
        kpt_idx = np.append(kpt_idx, 0)
    else:
        print(f"ERROR!. Approach {approach}. WTF?")
        exit()

    # removing visibility and sampling only desired keypoints
    n_instances = joint_data.shape[0]
    processed_features = joint_data[:, kpt_idx, 0:2]
    processed_features = processed_features.reshape(n_instances, -1)
    dim = processed_features.shape[-1]

    # substracting nose keypoint to enforce translation invariance
    ids_x = np.where(np.arange(dim) % 2 == 0)[0]
    ids_y = np.where(np.arange(dim) % 2 != 0)[0]
    noses_x, noses_y = processed_features[:,:1], processed_features[:,1:2]
    noses_x = np.repeat(noses_x, dim//2, axis=1)
    noses_y = np.repeat(noses_y, dim//2, axis=1)
    zero_idx = (processed_features == 0)
    processed_features[:,ids_x] = processed_features[:,ids_x] - noses_x
    processed_features[:,ids_y] = processed_features[:,ids_y] - noses_y
    processed_features[zero_idx] = 0


    if(normalize):
        epsilon = 1e-5
        norms = np.linalg.norm(processed_features, axis=1)[:,np.newaxis]
        norms[np.where(norms < epsilon)[0]] = epsilon  # for removing empty poses (norm = 0)
        processed_features = processed_features / norms
        norm = " normalized"
    else:
        norm = ""
    print(f"Processing {n_instances}{norm} pose vectors of dimensionality {dim}")

    return processed_features


def _create_graph(features, metric):
    """
    Creating a HNSW graph for knn retrieval
    """
    num_elements, embedding_dim = features.shape
    if(metric == "euclidean_distance"):
        space = "l2"
    elif(metric == "cosine_similarity"):
        space = "cosine"

    # creating hns graph object and fitting features
    m = 8              # minimum number of outgoing edges for each node
    ef = 1000           # increase query "volume" to make results more stable
    graph = hnswlib.Index(space=space, dim=embedding_dim)
    graph.init_index(max_elements=num_elements, ef_construction=ef, M=m)
    graph.set_ef(ef)
    graph.add_items(features, np.arange(num_elements))

    return graph


def fit_knn_structure(processed_features, data, params):
    """
    Fitting a knn structure (presumably a graph) to retrieve the most similar objects
    based on a certain distance. Sacve

    Args:
    -----
    processed_features: numpy array
        Array containing the data, already preprocessed and ready to be fit to the knn
    data: dictionary
        dictionary containing data and metadata from all desired datasets
    params: namespace
        Namespace containing the retrieval parameters (datasets, metric, ...)
    """

    if(params.approach != "full_body"):
        approach = f"approach_{params.approach}_"
    else:
        approach = ""
    cur_name = f"{os.path.basename(params.database_file)[:-4]}_"\
               f"metric_{params.metric}_norm_{approach}{params.normalize}.pkl"

    # setting up the knn hnsw graph
    knn = _create_graph(features=processed_features, metric=params.metric)

    # saving the hnswg
    knn_path = os.path.join(CONFIG["paths"]["knn_path"], f"graph_{cur_name}")
    knn.save_index(knn_path)

    # saving the joint data/metadata
    data_path = os.path.join(CONFIG["paths"]["knn_path"], f"data_{cur_name}")
    with open(data_path, "wb") as file:
        pickle.dump(data, file)

    # saving the joint processed_features
    data_path = os.path.join(CONFIG["paths"]["knn_path"], f"features_{cur_name}")
    with open(data_path, "wb") as file:
        pickle.dump(processed_features, file)

    return


if __name__ == "__main__":
    os.system("clear")
    args = process_arguments()
    data = load_data(database_file=args.database_file)
    processed_features = process_data(data=data, approach=args.approach,
                                      normalize=args.normalize)
    fit_knn_structure(processed_features=processed_features, data=data, params=args)

#
