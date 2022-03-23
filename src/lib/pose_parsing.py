"""
Methods for parsing the huma pose given the Heatmaps and PAFs

EnhancePoseEstimation/src/lib
@author: Angel Villar-Corrales 
"""

import os
import math
from operator import itemgetter

import numpy as np
import torch
import torch.nn.functional as F
from skimage.feature import peak_local_max

import lib.transforms as custom_transforms

# constants
SKELETON = None
BODY_PARTS_PAF_IDS = None


def get_max_preds_hrnet(scaled_heats, thr=0.1):
    """
    Obtaining joint positions and confidence values from heatmaps estimated by the HRNet model

    Args:
    -----
    scaled_heats: numpy array
        array containing the heatmaps predicted for a person bounding box (N, 17, 256, 192)

    Returns:
    --------
    preds: numpy array
        array containing the coordinates of the predicted joint (N, 17, 2)
    maxvals: numpy array
        array containing the value of the predicted joints (N, 17, 1)
    """

    batch_size = scaled_heats.shape[0]
    if(batch_size) == 0:
        return [], []
    num_joints = scaled_heats.shape[1]
    width = scaled_heats.shape[3]

    heatmaps_reshaped = scaled_heats.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask

    return preds, maxvals


def get_final_preds_hrnet(heatmaps, center, scale):
    """
    Obtaining the predicted keypoint coordinates and corresponding score from each
    heatmap. The coordinates are converted to the original image scale
    """

    coords, maxvals = get_max_preds_hrnet(heatmaps)

    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = heatmaps[n][p]
            px = int(np.floor(coords[n][p][0] + 0.5))
            py = int(np.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px] - hm[py-1][px]
                    ]
                )
                coords[n][p] = coords[n][p] + (np.sign(diff) * .25)

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = custom_transforms.transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals, coords


def keypoints_to_heatmaps(keypoints, shape=(192,256)):
    """
    """

    heatmaps = np.zeros((1,1,*shape))
    for i, kpt in enumerate(keypoints):
        kpt = kpt.long()
        heatmaps[0, 0, kpt[0], kpt[1]] = 1

    return heatmaps


def create_pose_entries(keypoints, max_vals=None, thr=0.1):
    """
    Creating pose objects from the detected joint-keypoints for the HRNet
    """

    if(len(keypoints) == 0):
        all_keypoints = []
    else:
        all_keypoints = np.array([(*item,1,1) for sublist in keypoints for item in sublist])
        idx = np.argwhere(all_keypoints==-1)
        all_keypoints[idx[:,0],:] = -1
        # filtering points that do not meet a confidence threshold
        if(max_vals is not None):
            idx = np.argwhere(max_vals[:,:,0] < thr)
            all_keypoints[idx[:,0] * 17 + idx[:,1], -1] = 0


    pose_entries = []
    pose_entry_size = 19

    for idx, cur_pose in enumerate(keypoints):
        pose_entry = np.ones(pose_entry_size) * -1
        for i, kpt in enumerate(cur_pose):
            if(kpt[0] != -1):
                pose_entry[i] = 17*idx + i
        pose_entry[-2] = 1
        pose_entry[-2] = len(np.where(pose_entry[:-2] !=- 1)[0])
        pose_entries.append(pose_entry)

    return pose_entries, all_keypoints


def create_pose_from_outputs(dets, keypoint_thr=0.1):
    """
    Creating pose vectors and keypoint lists given the outputs
    """
    n_dets = dets.shape[0]
    scaled_dets = F.interpolate(dets.clone(), (256, 192), mode="bilinear", align_corners=True)
    keypoint_coords, max_vals = get_max_preds_hrnet(scaled_dets.cpu().numpy())

    pose_entries, all_keypoints = create_pose_entries(keypoint_coords, max_vals,
                                                      thr=keypoint_thr)
    all_keypoints = [all_keypoints[:, 1], all_keypoints[:, 0],
                     all_keypoints[:, 2], all_keypoints[:, 3]]
    all_keypoints = np.array(all_keypoints).T


    return pose_entries, all_keypoints


# OPENPOSE
def extract_maxima_coords(img, pad=5, min_distance=1, thr=0.1):
    """
    Extracting the maxima from the input image. Used for selecting the most probable
    joint coordinate out from a heatmap
    """

    # shaping a normalizing
    padded_img = np.pad(img, pad_width=[(pad,pad),(pad,pad)], mode='constant', constant_values=0)
    if(np.max(padded_img) < 0.1):
        return [[], []], []
    padded_img = padded_img / np.max(padded_img)

    # thresholding
    padded_img[padded_img < thr] = 0

    # peak-picking the keypoints
    coords = peak_local_max(padded_img, min_distance=min_distance, indices=True)
    if(len(coords) > 0):
        coords = np.array(coords)
        vals = [padded_img[coord[0],coord[1]] for coord in coords]
        coords = coords - pad
    else:
        # coords = [[-1, -1]]
        # vals = [-1]
        coords = [[], []]
        vals = []

    return coords, vals


# OPENPOSE
def extract_joins_heatmap(heatmaps, keypoint_num=18, pad=5, min_distance=5, thr=0.1, label=False):
    """
    Extracting the keypoints for the joins from the predicted heatmaps

    Args:
    -----
    heatmap: numpy array
        matrices corresponding to the heatmaps
    keypoint_num: integer
        number of keypoints to detect
    pad: integer
        number of pixels to pad the image in every direction prior to detection
    min_distance: integer
        minimum distance in pixels between detected keypoints
    threshold: float
        pixels with values below the threshold will be set to zero
    label: boolean
        For debugging purposes. If True, extract joints from label heatmaps

    Returns:
    --------
    coords: numpy array
        binary matrix with True values in the pixels containing a join
    keypoints: list
        list containing the coordinates of each of the keypoints
    """

    # this method only works for batch size of one
    if(len(heatmaps.shape)==4 and heatmaps.shape[0]!=1):
        raise ValueError("Error. This method only works for a batch size of one...")
    elif(len(heatmaps.shape)==4 and heatmaps.shape[0]==1):
        heatmaps = heatmaps[0,:]

    # extracting keypoint coords
    keypoints = []
    n_keypoint = 0
    for idx in range(heatmaps.shape[0]-1):
        if(label):
            coords, vals = extract_maxima_coords(1-heatmaps[idx,:], thr=thr)
        else:
            coords, vals = extract_maxima_coords(heatmaps[idx,:], thr=thr)
        cur_keypoints = []
        for i in range(len(vals)):
            cur_keypoints.append([coords[i][0], coords[i][1], vals[i], n_keypoint])
            n_keypoint = n_keypoint + 1
        keypoints.append(cur_keypoints)

    # shaping a normalizing
    heatmap = 1 - heatmaps[-1]
    padded_heatmap = np.pad(heatmap, pad_width=[(pad,pad),(pad,pad)], mode='constant', constant_values=0)
    padded_heatmap = padded_heatmap / np.max(padded_heatmap)

    # thresholding
    padded_heatmap[padded_heatmap < thr] = 0

    coords = peak_local_max(padded_heatmap, min_distance=min_distance, indices=False)  # peak-picking the keypoints
    coords = coords[pad:-pad, pad:-pad]

    return coords, keypoints


# OPENPOSE
def group_keypoints(predicted_keypoints, predicted_pafs, pose_entry_size=20, min_paf_score=0.05,
                    thr_ratio=0.8, debug=False):
    """
    Combining the predicted keypoints integrating over the PAF scores
    Based on the implementation by: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

    Args:
    -----
    predicted_keypoints: list
        list containing the keypoints detected in the image
    predicted_pafs: numpy array (N,H,W)
        Array containing all PAFs detected from the image
    pose_entry_size: integer
        number of elemnts in a pose object (points_idx (x18), score, num_points)
    min_paf_score: float
        minimum score in the PAF to be considered a limb
    thr_ratio: float
        percentage of points between keypoints that must fulfill the PAF score condition
        for the limb to be considered as a feasible connection

    Returns:
    --------

    """

    # this method only works for batch size of one
    if(len(predicted_pafs.shape)==4 and predicted_pafs.shape[0]!=1):
        raise ValueError("Error. This method only works for a batch size of one...")
    elif(len(predicted_pafs.shape)==4 and predicted_pafs.shape[0]==1):
        predicted_pafs = predicted_pafs[0,:]

    pose_entries = []
    all_keypoints = np.array([item for sublist in predicted_keypoints for item in sublist])

    for part_id, limb_pair in enumerate(BODY_PARTS_PAF_IDS):

        # obtaining init and end keypoints for a given limb
        current_paf = predicted_pafs[limb_pair,:]
        a_idx, b_idx = SKELETON[part_id][0], SKELETON[part_id][1]
        a_list, b_list = predicted_keypoints[a_idx], predicted_keypoints[b_idx]
        num_kpts_a = len(a_list)
        num_kpts_b = len(b_list)

        # no keypoints for such body part
        if num_kpts_a == 0 and num_kpts_b == 0:
            continue

        # body part has just 'b' keypoints
        elif num_kpts_a == 0:
            for j in range(len(b_list)):
                num = 0
                for k in range(len(pose_entries)):   # check if keypoint is already in a pose
                    if (pose_entries[k][b_idx] == b_list[j][3]):
                        num = num + 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[b_idx] = b_list[j][3]     # keypoint idx
                    pose_entry[-1] = 1                # num keypoints in pose
                    pose_entries.append(pose_entry)
            continue

        # body part has just 'a' keypoints
        elif num_kpts_b == 0:
            for j in range(len(a_list)):
                num = 0
                for k in range(len(pose_entries)):
                    if (pose_entries[k][a_idx] == a_list[j][3]):
                        num = num + 1
                        continue
                if num == 0:
                    # print("Append in hidden B")
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[a_idx] = a_list[j][3]
                    pose_entry[-1] = 1
                    pose_entries.append(pose_entry)
            continue

        # computing connections between keypoints
        connections = []
        for j in range(len(a_list)):
            kpt_a = np.array(a_list[j])[:2]
            for k in range(len(b_list)):
                kpt_b = np.array(b_list[k])[:2]

                # computing midpoint and vector joining two keypoints
                mid_point = [None, None]
                mid_point = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                             int(round((kpt_a[1] + kpt_b[1]) * 0.5)))

                vector = np.array([kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]])
                vector_norm = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
                if vector_norm < 1e-7:  # same kepyoint, no vector
                    continue
                vector = vector / vector_norm

                # computing a preliminary score for the candidate limb based on the PAF
                paf_or = np.array(np.logical_or(current_paf[0,:], current_paf[1, :]))
                cur_point_score = (vector[0] * paf_or[mid_point[0], mid_point[1]] +
                                   vector[1] * paf_or[mid_point[0], mid_point[1]])
                # cur_point_score = (vector[0] * current_paf[0, mid_point[0], mid_point[1]] +
                                   # vector[1] * current_paf[1, mid_point[0], mid_point[1]])

                height_n = current_paf.shape[1] // 2
                success_ratio = 0
                ratio = 0
                point_num = 15  # number of points considered for integration over paf
                # if the preliminary score is optimistic, we integrate over the limb PAF
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    y, x = linspace2d(kpt_a, kpt_b, n=point_num)
                    for point_idx in range(point_num):
                        px = int(round(x[point_idx]))
                        py = int(round(y[point_idx]))

                        # paf = current_paf[:, py, px]
                        paf = [paf_or[py,px]] * 2

                        cur_point_score = vector[0] * paf[0] + vector[1] * paf[1]
                        cur_point_score = np.abs(cur_point_score)
                        if cur_point_score > min_paf_score:
                            passed_point_score += cur_point_score
                            passed_point_num += 1
                    success_ratio = passed_point_num / point_num
                    ratio = 0
                    if passed_point_num > 0:
                        ratio = passed_point_score / passed_point_num
                    ratio += min(height_n / vector_norm - 1, 0)
                if ratio > 0 and success_ratio > thr_ratio:
                    connections.append([j, k, ratio])

        # sorting the feasible connections by integration score
        if len(connections) > 0:
            # connections = sorted(connections, reverse=True, key=itemgetter(12))
            connections = sorted(connections, reverse=True)

        # picking best connections among all valid connections in a greedy fashion
        num_connections = min(len(a_list), len(b_list))
        has_kpt_a = np.zeros(len(a_list))
        has_kpt_b = np.zeros(len(b_list))

        filtered_connections = []
        for row in range(len(connections)):
            if len(filtered_connections) == num_connections:
                break
            j, k, cur_point_score = connections[row][0:3]
            if not has_kpt_a[j] and not has_kpt_b[k]:
                filtered_connections.append([a_list[j][3], b_list[k][3], cur_point_score])
                has_kpt_a[j] = 1
                has_kpt_b[k] = 1
        connections = filtered_connections
        if len(connections) == 0:
            continue

        # fitting the detected connections into the pose skeleton objects
        if(part_id == 0):
            pose_entry = np.ones(pose_entry_size) * -1
            pose_entries = [pose_entry for _ in range(len(connections))]
            for j in range(len(connections)):
                pose_entries[j][SKELETON[0][0]] = connections[j][0]
                pose_entries[j][SKELETON[0][1]] = connections[j][1]
                pose_entries[j][-1] = 2
                pose_entries[j][-2] = np.sum(all_keypoints[connections[j][0:2], 2]) + connections[j][2]
        elif part_id == 17 or part_id == 18:
            kpt_a_id = SKELETON[part_id][0]
            kpt_b_id = SKELETON[part_id][1]
            for j in range(len(connections)):
                for k in range(len(pose_entries)):
                    if pose_entries[k][kpt_a_id] == connections[j][0] and pose_entries[k][kpt_b_id] == -1:
                        pose_entries[k][kpt_b_id] = connections[j][1]
                    elif pose_entries[k][kpt_b_id] == connections[j][1] and pose_entries[k][kpt_a_id] == -1:
                        pose_entries[k][kpt_a_id] = connections[j][0]
            continue
        else:
            kpt_a_id = SKELETON[part_id][0]
            kpt_b_id = SKELETON[part_id][1]
            for j in range(len(connections)):
                num = 0
                for k in range(len(pose_entries)):
                    if(pose_entries[k][kpt_a_id] == connections[j][0]):
                        pose_entries[k][kpt_b_id] = connections[j][1]
                        num += 1
                        pose_entries[k][-1] += 1
                        pose_entries[k][-2] += all_keypoints[connections[j][1], 2] + connections[j][2]
                if num == 0:
                    # print("Appending due to connections")
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[j][0]
                    pose_entry[kpt_b_id] = connections[j][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[j][0:2], 2]) + connections[j][2]
                    pose_entries.append(pose_entry)

    pose_entries_merged = merge_poses(pose_entries)

    # Removing Poses with less than 3 keypoints or with a too low score/joint
    # for i in range(len(pose_entries)):
    #     print(f"Pose {i+1}")
    #     print(pose_entries[i][-1])
    #     print(pose_entries[i][-2])
    #     print("\n")
    #
    # for i in range(len(pose_entries_merged)):
    #     print(f"Pose merged {i+1}")
    #     print(pose_entries_merged[i][-1])
    #     print(pose_entries_merged[i][-2])
    #     print("\n")

    filtered_entries = []
    for i in range(len(pose_entries_merged)):
        if pose_entries_merged[i][-1] < 3 or (pose_entries_merged[i][-2] / pose_entries_merged[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries_merged[i])
    pose_entries_merged = np.asarray(filtered_entries)

    if(debug):
        return pose_entries_merged, all_keypoints, pose_entries


    return pose_entries_merged, all_keypoints


# OPENPOSE
def merge_poses(poses):
    """
    Iterating all pose objects, merging the ones corresponding to the same person
    and removing redundant poses

    Args:
    -----
    poses: list
        list with all the detected pose objectd
    """

    merged_poses = []

    # merging poses
    for i, pose_a in enumerate(poses):
        cur_merged_pose = np.copy(pose_a)
        for j, pose_b in enumerate(poses):
            for k in range(len(pose_a) - 2):
                if(cur_merged_pose[k]==pose_b[k] and cur_merged_pose[k]!=-1):
                    cur_merged_pose = merge(cur_merged_pose, pose_b)
        merged_poses.append(cur_merged_pose)

    # removing duplicates
    final_poses = []
    for i, pose_a in enumerate(merged_poses):
        if(i==0):
            final_poses.append(pose_a)
        append = True
        for j, pose_b in enumerate(final_poses):
            cur_pose_a = np.copy(pose_a)
            idx = np.where(cur_pose_a == -1)[0]
            cur_pose_a[idx] = -2
            idx = np.where(cur_pose_a[:-2]==pose_b[:-2])[0]
            n_equal_joints = len(idx)
            if(n_equal_joints > 3):
                append = False
        if(append):
            final_poses.append(pose_a)

    return final_poses


# OPENPOSE
def merge(pose_a, pose_b):
    """
    Merging two pose objects that belong to the same person.

    Args:
    -----
    pose_a, pose_b: numpy array
        arrays corresponding to the poses to merge
    """

    pose = [np.maximum(pose_a[i], pose_b[i]) for i in range(len(pose_a) - 2)]
    n_parts = len(np.where(np.array(pose) != -1)[0])
    pose.append(pose_a[-2] + pose_b[-2])
    pose.append(n_parts)
    merged_pose = np.array(pose)

    return merged_pose


# OPENPOSE
def linspace2d(start, stop, n=10):
    """
    Sampling N points between joints to integrate over the PAFs

    Args:
    -----
    start, stop: tuple (x,y)
        coordinates of the joints to integrate between
    N: integer
        number of points to sample between joints

    Returns:
    --------
    line: numpy array
        array containing N points sampled equidistantly from the line that joins the two keypoints
    """

    points = 1 / (n - 1) * (stop - start)
    line = points[:, None] * np.arange(n) + start[:, None]

    return line


#
