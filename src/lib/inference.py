"""
Methods for performing forward pass and inference through the model

@author: Angel Villar-Corrales 
"""

import numpy as np
import torch
import torch.nn.functional as F

import data.data_processing as data_processing
import lib.bounding_box as bbox
import lib.transforms as custom_transforms
import CONSTANTS


def forward_pass(model, img, model_name, device=None, flip=False):
    """
    Computing a forward pass throught the model, given the name of the architecture
    """

    if(model_name == "OpenPoseVGG"):
        pred_pafs, pred_heatmaps = model(img)
        return pred_pafs, pred_heatmaps

    elif(model_name == "OpenPose"):
        stages_output = model(img)
        pred_pafs = stages_output[-1]
        pred_heatmaps = stages_output[-2]
        return pred_pafs, pred_heatmaps

    elif(model_name == "HRNet"):
        # forward pass for input image
        output = model(img)
        # forward pass for flipped image
        if(flip is True):
            flipped_img = img.flip(3)
            output_flipped = model(flipped_img)
            output_flipped = custom_transforms.flip_back(output_flipped, CONSTANTS.FLIP_PAIRS)
            output_flipped = output_flipped.to(device)
            output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5 # averaging both predicitions
        return output

    else:
        raise NotImplementError("Wrong model name given. Only ['OpenPose'," +\
                                " 'OpenPoseVGG', 'HRNet'] are allowed")

    return


def multiscale_inference(model, imgs, scales, image_size, device, model_name, **kwargs):
    """
    Calling the corresponding multiscale_inference method given the current model architecture
    """

    if("OpenPose" in model_name):
        avg_heatmaps, avg_paf = multiscale_inference_open_pose(model, imgs, scales, image_size,
                                                               device, model_name, **kwargs)
        bboxes = torch.empty(1)

    elif(model_name == "HRNet"):
        # patch size is hardcoded to be (256, 192)
        avg_heatmaps, bboxes = multiscale_inference_hrnet(model, imgs, scales, (256,192),
                                                          device, model_name, **kwargs)
        avg_paf = torch.empty(1)

    return avg_heatmaps, avg_paf, bboxes


# DEPRECATED
def multiscale_inference_hrnet(model, imgs, scales, image_size, device,
                               model_name, detector, **kwargs):
    """
    Performing a forward pass through the HRNet model for different scaling factors
    and averaging the predicted heatmaps
    """

    batch_size = imgs.shape[0]
    imgs_f = imgs #* 256 + 128
    # bboxes, labels, scores = detector.predict(imgs_f, visualize=True)
    predictions = detector(imgs_f.to(device).float())
    bboxes, labels, scores = bbox.bbox_filtering(predictions, filter=1)

    detections = bbox.get_detections(imgs, bboxes)

    avg_heatmaps = torch.zeros((len(detections), 17, image_size[0], image_size[1])).to(device)

    if(len(detections) == 0):
        return avg_heatmaps, bboxes

    detections = detections.to(device).float()
    dets_per_img = [0] + [len(bb) for bb in bboxes]

    for ratio in scales:
        target_size = np.round(ratio*np.array(image_size)).astype(int).tolist()
        scaled_detections = F.interpolate(detections, target_size,
                                          mode="bilinear", align_corners=True)

        predicted_heats = forward_pass(model=model, img=scaled_detections, model_name=model_name)
        scaled_heats = F.interpolate(predicted_heats.clone().detach(), image_size,
                                     mode="bilinear", align_corners=True).to(device)

        if(avg_heatmaps is None):
            avg_heatmaps = scaled_heats
        else:
            avg_heatmaps = avg_heatmaps + scaled_heats

    avg_heatmaps = avg_heatmaps / len(scales)

    return avg_heatmaps, bboxes


def multiscale_inference_open_pose(model, imgs, scales, image_size, device, model_name, **kwargs):
    """
    Performing a forward pass through the OpenPose model for different scaling factors
    and averaging the predicted heatmaps and pafs
    """

    batch_size = imgs.shape[0]
    avg_heatmaps = np.zeros((batch_size, 19, image_size, image_size), dtype=np.float32)
    avg_pafs = np.zeros((batch_size, 38, image_size, image_size), dtype=np.float32)
    avg_heatmaps = torch.Tensor(avg_heatmaps).to(device)
    avg_pafs = torch.Tensor(avg_pafs).to(device)

    for ratio in scales:

        scaled_img = F.interpolate(imgs, int(round(ratio*image_size)), mode="bilinear", align_corners=True)

        pred_pafs_, pred_heatmaps_ = forward_pass(model=model, img=scaled_img, model_name=model_name)

        scaled_pafs = F.interpolate(pred_pafs_.clone().detach(), image_size, mode="bilinear", align_corners=True).to(device)
        scaled_hms = F.interpolate(pred_heatmaps_.clone().detach(), image_size, mode="bilinear", align_corners=True).to(device)

        avg_heatmaps += scaled_hms
        avg_pafs += scaled_pafs

    avg_heatmaps = avg_heatmaps / len(scales)
    avg_pafs = avg_pafs / len(scales)

    # upscaling model predictions to match original size
    avg_heatmaps, avg_pafs = data_processing.upscale_heatmaps_pafs(avg_heatmaps, avg_pafs,
                                                                     target_size=image_size)


    return avg_heatmaps, avg_pafs


# DEPRECATED
def box_to_center_scale(bbox, width, height):
    """
    Centering and converting a person bounding box to the required scale

    Args:
    -----
    bbox : list
        array containing the coordinates with the predicted human bounding boxes
        (y_{min}, x_{min}, y_{max}, x_{max})
    width, height : int
        size of the bounding boxes

    Returns:
    --------
    center: tuple
        coordinates (x,y) of the center of the box wrt the image
    scales: tuple
        Scale factor for the x and y coordinates used to generate the bounding box
    """

    box_width = box[3] - box[1]
    box_height = box[2] - box[0]

    # computing
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


#
