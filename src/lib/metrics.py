"""
Methods for computing predictions and evaluation metrics, incuding OKS Precision
from the COCO api and accuracy based on distance

EnhancePoseEstimation/src/lib
@author: Angel Villar-Corrales 
"""

import os
import json
from collections import defaultdict
from collections import OrderedDict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import data.data_processing as data_processing
import data.custom_transforms as custom_transforms
import lib.utils as utils
import lib.nms as nms_lib
import lib.bounding_box as bbox
import lib.pose_parsing as pose_parsing
from CONFIG import CONFIG
import CONSTANTS


#######################################
####### POSE RETRIEVAL METRICS ########
#######################################

def score_retrievals(label, retrievals):
    """
    Evaluating the current retrieval experiment

    Args:
    -----
    label: string
        label corresponding to the query
    retrivals: list
        list of strings containing the ranked labels corresponding to the retrievals
    tot_labels: integer
        number of images with the current label. We need this to compute recalls
    """

    retrievals = retrievals[1:] # we do not account rank-0 since it's self-retrieval
    relevant_mask = np.array([1 if r==label else 0 for r in retrievals])
    num_relevant_retrievals = np.sum(relevant_mask)

    if(num_relevant_retrievals == 0):
        print(label)
        metrics = {
            "label": label,
            "p@1": -1,
            "p@5": -1,
            "p@10": -1,
            "p@rel": -1,
            "mAP": -1,
            "r@1": -1,
            "r@5": -1,
            "r@10": -1,
            "r@rel": -1,
            "mAR": -1
        }
        return metrics

    # computing precision based metrics
    precision_at_rank = np.cumsum(relevant_mask) / np.arange(1, len(relevant_mask) + 1)
    precision_at_1 = precision_at_rank[0]
    precision_at_5 = precision_at_rank[4]
    precision_at_10 = precision_at_rank[9]
    precision_at_rel = precision_at_rank[num_relevant_retrievals - 1]
    average_precision = np.sum(precision_at_rank * relevant_mask) / num_relevant_retrievals

    # computing recall based metrics
    recall_at_rank = np.cumsum(relevant_mask) / num_relevant_retrievals
    recall_at_1 = recall_at_rank[0]
    recall_at_5 = recall_at_rank[4]
    recall_at_10 = recall_at_rank[9]
    recall_at_rel = recall_at_rank[num_relevant_retrievals - 1]
    average_recall = np.sum(recall_at_rank * relevant_mask) / num_relevant_retrievals

    metrics = {
        "label": label,
        "p@1": precision_at_1,
        "p@5": precision_at_5,
        "p@10": precision_at_10,
        "p@rel": precision_at_rel,
        "mAP": average_precision,
        "r@1": recall_at_1,
        "r@5": recall_at_5,
        "r@10": recall_at_10,
        "r@rel": recall_at_rel,
        "mAR": average_recall
    }

    return metrics


#######################################
####### POSE SIMILARITY METRICS #######
#######################################
def confidence_score(query, pose_db, confidence):
    """
    Computing the confidence score for pose similarity. This metric weights the distance
    between keypoints with the confidence with which each point was detected

    Args:
    -----
    query, pose_db: numpy array
        pose vectors for the query and database image
    confidence: numpy array
        vector with the confidence  with which each query keypoint was detected
    """

    # normalizing with the sum of confidences so metric is bounded by 1
    confidence = confidence / np.sqrt(np.sum(np.power(confidence,2)))
    norm = 1 / (np.sum(confidence))
    weighted_scores = np.sqrt(np.sum(confidence * np.power(query - pose_db, 2)))
    confidence_score = norm * weighted_scores

    return confidence_score


def oks_score(query, pose_db, approach):
    """
    Computing the object keypoint similarity between two poses. Metric inspired by
    flow-based person tracking in videos

    Args:
    -----
    query, pose_db: numpy array
        pose vectors for the query and database image
    """

    # defining and normalizing variance of the gaussians for each keypojnt
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,
                       .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    if(len(query) == 26):
        kpt_idx = np.arange(5, 17)  # from shoulders to ankles
        kpt_idx = np.append(kpt_idx, 0)
    elif(len(query) == 34):
        kpt_idx = np.arange(17)  # all keypoints
    else:
        kpt_idx = np.arange(5, 13)  # from shoulders to hips
        kpt_idx = np.append(kpt_idx, 0)
    sigmas = sigmas[kpt_idx]

    square_dists = [(query[2*i] - pose_db[2*i])**2 + (query[2*i+1] - pose_db[2*i+1])**2
                    for i in range(len(query) // 2)]
    exponent = square_dists / (np.power(sigmas, 2) * 2)
    oks = np.sum( np.exp(-1 * exponent) ) / (len(query) // 2)

    oks = 1 - oks  # unlike distance, the larger oks the better, so we do this :)

    return oks


#######################################
######### COCO EVAL METRICS ###########
#######################################

def compute_precision(preds_file, labels_file, summarize=False):
    """
    Computing the precission and recall values for the processed images
    using the coco eval api.

    Args:
    -----
    preds_file, labels_file: string
        path to the files with the predictions and with the labels respectively

    Returns:
    --------
    stats: list
        list containing the precision and recall values for each of the coco metrics
    """

    coco_lbl = COCO(labels_file)
    coco_det = coco_lbl.loadRes(preds_file)

    cocoEval = COCOeval(coco_lbl, coco_det, "keypoints")
    cocoEval.params.useSegm = None

    # extracting image ids so we only compute metrics on the evaluated images
    cocoEval._prepare()
    preds = utils.load_predictions(preds_file)
    image_ids = [pred["image_id"] for pred in preds]
    cocoEval.params.imgIds  = image_ids

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats = cocoEval.stats

    return stats


def generate_submission_hrnet(all_preds, all_bboxes, image_ids, preds_file, name=False):
    """
    Generating a file withe the predictions to then perform the evaluation with CocoEval

    Args:
    -----
    all_preds: np array
        Array containing the coordinates of the predicted bboxes (n_imgs, 17, 3)
    all_bboxes: np array
        Array containing the bbox information about the instances (n_imgs, 6)
    image_idx: list
        List with the ids of the images to evaluate. It is used to match predicted poses
        with the ground-truth from CocoEval
    preds_file: string
        path to the json file where predictins are stored
    name: bool
        if True, image_id is extracted from image_name (only for MS-COCO and Styled-COCO)
    """

    all_preds = np.concatenate(all_preds, axis=0)
    all_bboxes = np.concatenate(all_bboxes, axis=0)

    if(name):
        image_ids = [int(name[-16:-4]) for name in image_ids]

    kpts_per_person = []
    for idx, kpt in enumerate(all_preds):
        kpts_per_person.append({
            'keypoints': kpt,
            'center': all_bboxes[idx][0:2],
            'scale': all_bboxes[idx][2:4],
            'area': all_bboxes[idx][4],
            'score': all_bboxes[idx][5],
            'image': image_ids[idx]
        })
    # image x person x (keypoints)
    kpts_per_img = defaultdict(list)
    for kpt in kpts_per_person:
        kpts_per_img[kpt['image']].append(kpt)

    # rescoring and oks nms
    num_joints = 17
    in_vis_thr = 0.2
    oks_thr = 0.9
    oks_nmsed_kpts = []

    for img in kpts_per_img.keys():
        img_kpts = kpts_per_img[img]
        for person_data in img_kpts:
            box_score = person_data['score']
            kpt_score = 0
            valid_num = 0
            for n_joint in range(0, num_joints):
                joint_score = person_data['keypoints'][n_joint][2]
                if joint_score > in_vis_thr:
                    kpt_score = kpt_score + joint_score
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            # rescoring
            person_data['score'] = kpt_score * box_score

        keep = nms_lib.oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thr)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[k] for k in keep])

    # loading previous results, merging with new ones and saving
    results = data_processing.convert_keypoints_to_coco_format(oks_nmsed_kpts, preds_file)
    with open(preds_file, 'w') as f:
        json.dump(results, f)

    return


# DEPRECATED
def compute_predictions(model_name, **kwargs):
    """
    Calling the corresponding precision method given the model name
    """

    if("OpenPose" in model_name):
        cur_results = compute_predictions_openpose(**kwargs)
    elif(model_name == "HRNet"):
        cur_results = compute_predictions_hrnet(**kwargs)

    return cur_results


# REVIEWING: PROBABLY DEPRECATED
def compute_predictions_hrnet(heatmaps, bboxes, metadata, cur_size=400, **kwargs):
    """
    Extrancting pose predictions from the detected heatmaps and bounding boxes

    Args:
    -----
    heatmaps: numpy array
        np matrices corresponding to the heatmaps estimated from the bounding boxes
    bboxes: list
        list with the bounding box coordinates [[y0, x0, y1, x1], ...]
    metadata: dictionary
        metadata corresponding to the sampled image. It includes the image name/path and the
        original image size to undo the resizing
    """

    img_names, img_shapes = metadata["image_name"],  np.array(metadata["image_shape"])
    dets_per_img = [0] + [len(bb) for bb in bboxes]

    # estimating keypoints for each bounding box
    keypoints, _ = pose_parsing.get_max_preds_hrnet(heatmaps, thr=0.1)
    reshaped_keypoints = bbox.bbox_to_image_keypoints(np.copy(keypoints), bboxes,
                                                      height=256, width=192)

    # creating pose objects and grouping poses per image
    pose_entries, all_keypoints = pose_parsing.create_pose_entries(reshaped_keypoints)
    pose_entries_per_img = [pose_entries[dets_per_img[i]:dets_per_img[i] + dets_per_img[i+1]]
                            for i in range(len(dets_per_img[:-1]))]

    # reshaping the points to match coco eval format
    cur_results = None
    for i, pose in enumerate(pose_entries_per_img):
        img_shape, img_name = img_shapes[i], img_names[i]
        coco_keypoints_, scores = data_processing.convert_to_coco_format(pose, all_keypoints)
        coco_keypoints = data_processing.resize_inference(coco_keypoints_, original_size=img_shape,
                                                          cur_size=cur_size)
        cur_results = data_processing.convert_to_submission_dictionary(coco_keypoints, scores,
                                                                       img_name, cur_results)

    return cur_results


# DEPRECATED
def compute_predictions_openpose(heatmaps, pafs, metadata, cur_size=400, thr=0.5,
                                 thr_ratio=0.8, **kwargs):
    """
    Extrancting pose predictions from the detected heatmaps and pafs

    Args:
    -----
    heatmaps, pafs: numpy array
        np matrices corresponding to the estimated heatmaps and pafs
    metadata: dictionary
        metadata corresponding to the sampled image. It includes the image name/path and the
        original image size to undo the resizing
    """

    if(isinstance(metadata["image_name"],list)):
        img_name, img_shape = metadata["image_name"][0],  np.array(metadata["image_shape"][0])
    else:
        img_name, img_shape = metadata["image_name"],  np.array(metadata["image_shape"])

    # extracting and matching keypoints to poses
    _, keypoints = pose_parsing.extract_joins_heatmap(heatmaps, min_distance=1, thr=thr)
    pose_entries, all_keypoints = pose_parsing.group_keypoints(keypoints, pafs, min_paf_score=0.05,
                                                               thr_ratio=thr_ratio)

    # reshaping the points to match coco eval format
    coco_keypoints_, scores = data_processing.convert_to_coco_format(pose_entries, all_keypoints)
    coco_keypoints = data_processing.resize_inference(coco_keypoints_, original_size=img_shape,
                                                      cur_size=cur_size)
    cur_results = data_processing.convert_to_submission_dictionary(coco_keypoints, scores, img_name)

    return cur_results


def calc_dists(preds, target, normalize):
    """
    Measure the euclidean distance between  predictions and target for computing
    the accuracy

    Args:
    -----
    preds, target: numpy array
        Array with shape (B,n_joints,H,W) corresponding repectively to the model output and labels
    normalize: numpy array
        Normalization factors for prediction and target arrays

    Returns:
    --------
    dists: numpy array
        Array with shape (n_joints, B) with the distance between each pairs (pred, target)
    """

    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """
    Computing the percentage of points for which the distance to the target is below
    the distance threshold. Points with distance -1 (labeled as invisible) are not considered

    Args:
    -----
    dists: numpy array
        Array with shape (n_joints, B) with the distance between each pairs (pred, target)
    thr: float
        distance threshold
    """

    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    """
    Calculate accuracy according to PCK, but using ground truth heatmap rather than
    coordinate (x,y) locations directly.

    Args:
    -----
    outputs, target: torch tensor
        tensors with shape (B, n_joints, H, W) corresponding to the predicted heatmaps
        and the annotated heatmaps

    Returns:
    --------
    acc: float
        average accuracy accross all keypoints
    avg_acc: list
        average accuracy for each keypoint independently
    """

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = pose_parsing.get_max_preds_hrnet(output)
        target, _ = pose_parsing.get_max_preds_hrnet(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred



#
