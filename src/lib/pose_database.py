"""
Methods for processing the retrieval dabase fit with pose skeletons from all person
instanes in certain dataset, i.e., MS-COCO, Styled-COCO or Vases

@author: 
"""

import os
import pickle

import numpy as np
import hnswlib

from lib.logger import print_
from lib.metrics import oks_score, confidence_score
from CONFIG import CONFIG


def process_pose_vector(vector, approach, normalize=True):
    """
    Processing a pose matrix (17 keypoints, 3 features) into a pose vector to
    perform pose-based retrieval

    Args:
    -----
    vector: numpy array
        Array with shape (17,3) to convert into a pose vector for retrieval
    approach: string
        Approach (keypoints) used to measure similarity
    normalize: boolean
        If True, normalized pose vectors are used
    """

    # obtaining inidices for desired keypoints
    if(approach == "all_kpts"):
        kpt_idx = np.arange(17)  # all keypoints
    elif(approach == "full_body"):
        kpt_idx = np.arange(5, 17)  # from shoulders to hips
        kpt_idx = np.append(kpt_idx, 0)
    elif(approach == "upper_body"):
        kpt_idx = np.arange(5, 13)  # from shoulders to ankles
        kpt_idx = np.append(kpt_idx, 0)
    else:
        print(f"ERROR!. Approach {approach}. WTF?")
        exit()

    # removing visibility and sampling only desired keypoints
    if(len(vector.shape) > 1):
        processed_vector = vector[kpt_idx, 0:2].flatten()
    else:
        processed_vector = vector[kpt_idx]
    dim = processed_vector.shape[-1]

    # substracting nose keypoint to enforce translation invariance
    ids_x = np.where(np.arange(dim) % 2 == 0)[0]
    ids_y = np.where(np.arange(dim) % 2 != 0)[0]
    noses_x, noses_y = processed_vector[0], processed_vector[1]
    zero_idx = (processed_vector == 0)
    processed_vector[ids_x] = processed_vector[ids_x] - noses_x
    processed_vector[ids_y] = processed_vector[ids_y] - noses_y
    processed_vector[zero_idx] = 0

    if(normalize):
        norm = np.linalg.norm(processed_vector)
        epsilon = 1e-5
        norm = norm if norm > epsilon else 1e-5
        processed_vector = processed_vector / norm

    return processed_vector


def load_database(db_name, db_split="eval"):
    """
    Loading the pickled file with the database data

    Args:
    -----
    db_name: string
        name of the database to load

    Returns:
    --------
    database: dictionary
        dict containing the data from the retrieval database (img_name, annotations, ...)
    """

    db_path = CONFIG["paths"]["database_path"]
    pickle_path = os.path.join(db_path, f"database_{db_name}_{db_split}.pkl")
    with open(pickle_path, "rb") as file:
        database = pickle.load(file)

    if("data" in database.keys()):
        cur_db = database["data"]
    else:
        cur_db = database
    return cur_db


def load_knn(database_file):
    """
    Loading the prefit knn object for the current retrieval task

    Args:
    -----
    database_file: string
        name of the picle file containing the preprocess database

    Returns:
    --------
    knn: HNSW object
        HNSW graph used for the knn retrieval
    database: dictionary
        dict containing data and metadata from the database features
    features: numpy array
        data samples as fit into the knn object
    """

    # obtaining names and paths of the data and neighbors objects
    knn_dir = CONFIG["paths"]["knn_path"]
    name_mask = database_file[5:]  # removing the 'data_' head
    knn_path = os.path.join(knn_dir, f"graph_{name_mask}")
    data_path = os.path.join(knn_dir, f"data_{name_mask}")
    features_path = os.path.join(knn_dir, f"features_{name_mask}")

    # making sure those objects exist
    if(not os.path.exists(knn_path)):
        message = f"KNN path '{knn_path}' does not exists..."
        print_(message, message_type="error")
    if(not os.path.exists(data_path)):
        message = f"KNN data '{data_path}' does not exists..."
        print_(message, message_type="error")
    if(not os.path.exists(features_path)):
        message = f"KNN features '{features_path}' does not exists..."
        print_(message, message_type="error")

    # loading data
    with open(data_path, "rb") as file:
        database = pickle.load(file)
    with open(features_path, "rb") as file:
        features = pickle.load(file)
    dim = features.shape[-1]

    knn = hnswlib.Index(space='l2', dim=dim)
    knn.load_index(knn_path, max_elements=0)

    return knn, database, features


def get_neighbors_idxs(query, num_retrievals=10, approach="full_body",
                       retrieval_method="knn", penalization=None, **kwargs):
    """
    Iterating the database measuring distance from query to each dataset element
    and retrieving the elements with the smallest distance. Not Optimized

    Args:
    -----
    query: numpy array
        pose vector used as retrieval query
    k: integer
        number of elements to retrieve from the dataset
    approach: string
        strategy followed for the retrieval procedure
    penalization: string
        strategy followed to penalize the non-detected keypoints

    Returns:
    --------
    idx: list
        indices of the retrieved elements
    dist: list
        distances from the query to the database elements
    """

    if("scores" not in kwargs):
        confidence = np.ones(query.shape)
    else:
        confidence = kwargs["scores"]


    # kNN retrieval approach. Just giving the query to the fit kNN graph => O(log(N))
    if(retrieval_method == "knn"):
        assert "knn" in kwargs, "ERROR! 'knn' object was not given as parameter"
        knn = kwargs["knn"]
        idx, dists = knn.knn_query(query, k=num_retrievals)
        idx, dists = idx[0,:], dists[0,:]
        return idx, dists
    # defining method for computing metrics other than knn
    elif(retrieval_method == "cosine_similarity"):
        compute_metric = lambda x,y,z: 1 - np.dot(x,y)  # inverse cosine similarity
    elif(retrieval_method == "euclidean_distance"):
        compute_metric = lambda x,y,z: np.sqrt(np.sum(np.power(x - y, 2)))
    elif(retrieval_method == "manhattan_distance"):
        compute_metric = lambda x,y,z: np.sum(np.abs(x - y))
    elif(retrieval_method == "confidence_score"):
        compute_metric = lambda x,y,z: confidence_score(x, y, z)
    elif(retrieval_method == "oks_score"):
        # sigmas for gaussian kernel evaluated at each keypoint
        confidence = np.ones(query.shape)
        compute_metric = lambda x,y,z: oks_score(x, y, approach)
    else:
        print_(f"ERROR! Retrieval metric '{retrieval_method}' is not defined...")
        exit()

    # other methods require iterating the dataset => O(N)
    assert "database" in kwargs, "ERROR! 'database' object was not given as parameter"
    database = kwargs["database"]
    n_vectors, dims = database.shape

    if(penalization in ["mean", "max"]):
        penalization_value = get_penalization_metric(query=query, database=database,
                                                     penalization=penalization,
                                                     metric_func=compute_metric,
                                                     confidence=confidence)
    epsilon = 1e-5
    dists = []
    # print(query)
    for i, pose_vect in enumerate(database):

        # applying penalizations to ocluded points if necessary
        # ocluded points are assigned coordinate (0,0)
        if(penalization == "zero_coord"):
            cur_query = query
            cur_confidence = confidence
            cur_vect = pose_vect
        # removing kpts that are ocluded either in query or database item
        elif(penalization == "none"):
            cur_query, cur_confidence = np.copy(query), np.copy(confidence)
            cur_vect = np.copy(pose_vect)
            # obtainign idx of kpts that are 0 in query
            idx = np.where(np.abs(query) < epsilon)[0]
            cur_query[idx], cur_vect[idx], cur_confidence[idx] = 0, 0, 0
        # assigning mean/max value of metrics to ocluded points
        elif(penalization in ["mean", "max"]):
            cur_query, cur_confidence = np.copy(query), np.copy(confidence)
            cur_vect = np.copy(pose_vect)
            # obtainign idx of kpts that are 0 in (query AND NOT(db))
            idx = np.where((np.abs(query) < epsilon) & (np.abs(cur_vect) > epsilon))[0]
            cur_query[idx] = penalization_value
            cur_vect[idx], cur_confidence[idx] = 0, 0
        # computing specified metric and saving 'distance' results
        dist = compute_metric(cur_query, cur_vect, cur_confidence)
        dists.append(dist)

    idx = np.argsort(dists)[:num_retrievals]
    dists = [dists[i] for i in idx]

    return idx, dists


def get_penalization_metric(query, database, metric_func, penalization="mean",
                            confidence=None, N=100):
    """
    Computing the mean or max distance between query and database

    Args:
    -----
    query, database: np arrays
        pose vectors corresponding to the query and the database
    metric_func: function
        function used to compute the metric between vectors
    penalization: string
        type of penalization to apply: ['mean', 'max']
    confidence: numpy array
        vector with the confidence  with which each query keypoint was detected
    N: integer
        number of database elements considered to compute penalization_value
    """

    assert penalization in ["mean", "max"]

    dists = []
    for i, cur_vect in enumerate(database):
        if(i==N):
            break
        dist = metric_func(query, cur_vect, confidence)
        dists.append(dist)

    if(penalization == "mean"):
        return np.mean(dists)
    elif(penalization == "max"):
        return np.max(dists)


#
