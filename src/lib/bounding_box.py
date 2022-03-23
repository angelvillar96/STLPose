"""
Methods for handling and processing bounding boxes from the FasterRCNN model

@author: Angel Villar-Corrales 
"""

import numpy as np
import torch
import torchvision
import torch.nn.functional as F


# TODO: This method needs to be reviewed
def get_detections(imgs, bboxes, height=256, width=192):
    """
    Extracting all reshaped cropped human predictions from the image given the bounding boxes

     Args:
    ----------
    imgs: numpy array
        images from which the humans were detected
    bbox : list
        list containing the coordinates of all bboxes
    width, height : int
        size of the bounding boxes
    """

    detections = []
    for i, img_bbs in enumerate(bboxes):
        cur_dets = []
        for j, bb in enumerate(img_bbs):
            det = reshape_detection(imgs[i,:], bb, height=height, width=width)
            cur_dets.append(det)
        if(len(cur_dets) == 0):
            continue
        cur_dets = torch.stack(cur_dets).squeeze()
        if(len(cur_dets.shape) == 3):
             cur_dets = cur_dets.reshape(1, *cur_dets.shape)
        detections.append(cur_dets)

    if(len(detections) > 0):
        detections = torch.cat(detections, axis=0)
    # if(len(detections.shape) == 3):
         # detections = detections.reshape(1, *detections.shape)

    return detections


# TODO: This method needs to be reviewed
def reshape_detection(img, bb, height=256, width=192, offset=0):
    """
    Reshaping the size of the bounding boxes to the required by HRNet

    Args:
    ----------
    img: numpy array
        image from which the humans where detected
    bbox : list
        array containing the coordinates with the corresponding human bounding box
        (y_{min}, x_{min}, y_{max}, x_{max})
    width, height : int
        size of the bounding boxes
    """


    bb = [int(round(b)) for b in bb]
    bb[0], bb[1] = bb[0] - offset, bb[1] - offset
    bb[2], bb[3] = bb[2] + offset, bb[3] + offset
    box_height = bb[2] - bb[0]
    box_width = bb[3] - bb[1]
    center = np.array([bb[0] + box_height/2, bb[1] + box_width/2]).astype(int)

    cropped_img = img[:, bb[0]:bb[2], bb[1]:bb[3]].reshape(1, 3, box_height, box_width)
    reshaped_img = F.interpolate(cropped_img.clone().detach(), (height, width), mode="bilinear",
                                 align_corners=True)

    return reshaped_img


# TODO: This method needs to be reviewed
def bbox_to_image_keypoints(pred_keypoints, list_bboxes, height=256, width=192, offset=0):
    """
    Shifting the detected keypoints from bounding-box coordinates to image coordinates

    Args:
    -----
    pred_keypoints: numpy array
        array containing the predicted keypoints for all bounding boxes
    list_bboxes: list
        list with the bounding box coordinates [[y0, x0, y1, x1], ...]
    height, width: integer
        standarized height and width of the bounding boxes

    Returns:
    --------
    reshaped_keypoints: numpy array
        array containing the coordinates os each keypoint for the images
    """

    reshaped_keypoints = []

    # in case no keypoint has been detected, we do not reshape anything
    bb_length = np.sum(np.array([len(bb) for bb in list_bboxes]))
    if(bb_length == 0):
        return pred_keypoints

    bboxes = np.concatenate([np.array(bb) for bb in list_bboxes if(len(bb) > 0)])
    # bboxes = np.array(list_bboxes).reshape(-1,4)

    no_kpt_idx = np.argwhere(pred_keypoints == -1)

    for i, bbox in enumerate(bboxes):

        x_0, y_0 = bbox[1] - offset, bbox[0] - offset
        original_height = bbox[2] - bbox[0] + 2*offset
        original_width = bbox[3] - bbox[1] + 2*offset
        height_ratio = original_height / height
        width_ratio = original_width / width

        pred_keypoints[i,:,0] = pred_keypoints[i,:,0] * height_ratio + y_0
        pred_keypoints[i,:,1] = pred_keypoints[i,:,1] * width_ratio + x_0
        reshaped_keypoints.append(pred_keypoints[i, :])

    reshaped_keypoints = np.array(reshaped_keypoints)
    reshaped_keypoints = np.round(reshaped_keypoints).astype(int)
    reshaped_keypoints[no_kpt_idx[:,0], no_kpt_idx[:,1], no_kpt_idx[:,2]] = -1

    return reshaped_keypoints



def bbox_filtering(predictions, filter_=1, thr=0.6):
    """
    Filtering predicitions in order to keep only the relevant bounding boxes #
    (people in our particular case)

    Args:
    -----
    predictions: list
        list containign a dictionary with all predicted bboxes, labels and scores:
            bbox: numpy array
                Array of shape (N, 4) where N is the number of boxes detected.
                The 4 corresponds to x_min, y_min, x_max, y_max
            label: numpy array
                Array containing the ID for the predicted labels
            scores: numpy array
                Array containing the prediction confident scores
    filter: list
        list containing label indices that we wnat to keep
    thr: float
        score threshold for considering a bounding box
    """

    # import pdb.set_trace()
    filtered_bbox, filtered_labels, filtered_scores = [], [], []
    for pred in predictions:
        bbox, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
        cur_bbox, cur_labels, cur_scores = [], [], []
        for i, _ in enumerate(labels):
            if(labels[i] == filter_ and scores[i] > thr):
                aux = bbox[i].cpu().detach().numpy()
                reshaped_bbox = [aux[0], aux[1], aux[2], aux[3]]
                cur_bbox.append(reshaped_bbox)
                # cur_bbox.append(bbox[i].cpu().detach().numpy())
                cur_labels.append(labels[i].cpu().detach().numpy())
                cur_scores.append(scores[i].cpu().detach().numpy())
        # if(len(cur_bbox) == 0):
            # continue
        filtered_bbox.append(cur_bbox)
        filtered_labels.append(cur_labels)
        filtered_scores.append(cur_scores)


    return filtered_bbox, filtered_labels, filtered_scores


def bbox_nms(boxes, labels, scores, nms_thr=0.5):
    """
    Applying Non-maximum suppresion to remove redundant bounding boxes

    Args:
    -----
    boxes: list
        List of shape (N, 4) where N is the number of boxes detected.
        The 4 corresponds to x_min, y_min, x_max, y_max
    labels: list
        List containing the ID for the predicted labels
    scores: list
        List containing the prediction confident scores
    nms_thr: float
        threshold used for the NMS procedure
    """

    # import pdb
    # pdb.set_trace()
    boxes_, labels_, scores_ = [], [], []
    for i in range(len(boxes)):
        cur_boxes = np.array(boxes[i])
        cur_labels = np.array(labels[i])
        cur_scores = np.array(scores[i])
        idx = torchvision.ops.nms(boxes = torch.from_numpy(cur_boxes),
                                  scores = torch.from_numpy(cur_scores),
                                  iou_threshold = nms_thr)

        cur_boxes = np.array([cur_boxes[i] for i in idx])
        cur_labels = np.array([cur_labels[i] for i in idx])
        cur_scores = np.array([cur_scores[i] for i in idx])
        boxes_.append(cur_boxes)
        labels_.append(cur_labels)
        scores_.append(cur_scores)

    return np.array(boxes_), np.array(labels_), np.array(scores_)

#
