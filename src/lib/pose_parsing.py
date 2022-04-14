"""
Methods for parsing the huma pose given the Heatmaps and PAFs

@author: Angel Villar-Corrales
"""

import numpy as np
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


def keypoints_to_heatmaps(keypoints, shape=(192, 256)):
    """
    """

    heatmaps = np.zeros((1, 1, *shape))
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
        all_keypoints = np.array([(*item, 1, 1) for sublist in keypoints for item in sublist])
        idx = np.argwhere(all_keypoints == -1)
        all_keypoints[idx[:, 0], :] = -1
        # filtering points that do not meet a confidence threshold
        if(max_vals is not None):
            idx = np.argwhere(max_vals[:, :, 0] < thr)
            all_keypoints[idx[:, 0] * 17 + idx[:, 1], -1] = 0

    pose_entries = []
    pose_entry_size = 19

    for idx, cur_pose in enumerate(keypoints):
        pose_entry = np.ones(pose_entry_size) * -1
        for i, kpt in enumerate(cur_pose):
            if(kpt[0] != -1):
                pose_entry[i] = 17*idx + i
        pose_entry[-2] = 1
        pose_entry[-2] = len(np.where(pose_entry[:-2] != -1)[0])
        pose_entries.append(pose_entry)

    return pose_entries, all_keypoints


def create_pose_from_outputs(dets, keypoint_thr=0.1):
    """
    Creating pose vectors and keypoint lists given the outputs
    """
    scaled_dets = F.interpolate(dets.clone(), (256, 192), mode="bilinear", align_corners=True)
    keypoint_coords, max_vals = get_max_preds_hrnet(scaled_dets.cpu().numpy())

    pose_entries, all_keypoints = create_pose_entries(keypoint_coords, max_vals,
                                                      thr=keypoint_thr)
    all_keypoints = [all_keypoints[:, 1], all_keypoints[:, 0],
                     all_keypoints[:, 2], all_keypoints[:, 3]]
    all_keypoints = np.array(all_keypoints).T

    return pose_entries, all_keypoints


#
