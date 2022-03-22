"""
Methods for processing the annotations of the coco dataset. The methods in this file
are used to generate keypoint-heatmaps and Part Affinity Fields (PAFs) from the
original COCO annotation

EnhancePoseEstimation/src/data
@author: 
    Adapted from Daniil Osokin:
        https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""

import time

import numpy as np
import torch.nn.functional as F
import pycocotools

from CONSTANTS import REORDER_MAP

"""
0  - Nose         1  - Left Eye       2  - Right Eye        3  - Left Ear
4  - Right Ear    5  - Left Shoulder  6  - Right Shoulder   7  - Left Elbow
8  - Right elbow  9  - Left Hand      10 - Right hand       11 - Left Hip
12 - Right Hip    13 - Left Knee      14 - Right Knee       15 -  Left Ankle
16 - Right Ankle  17 - Neck (midpoint between shoulders)
"""

KERNEL = None
BODY_PARTS_KPT_IDS = None

TO_COCO_MAP = None
SKIP_NECK = False

def timimg(method):
    """
    Decorator (@timing) for measuring the execution time of a method
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % \
         (method.__name__, (te - ts)*1000))
        return result
    return timed


def reorder_keypoints_open_pose(target):
    """
    Reordering the keypoints to match the expected input of the Lightweight
    OpenPose pretrained model
    """

    for n in range(len(target)):
        keypoints = target[n]['keypoints']
        reordered_keypoints = []
        for idx in REORDER_MAP:
            reordered_keypoints = reordered_keypoints + keypoints[3*idx: 3*(idx+1)]
        reordered_keypoints = reordered_keypoints[:3] + keypoints[-3:] + reordered_keypoints[3:]
        target[n]['keypoints'] = reordered_keypoints

    return target


def convert_keypoints_to_coco_format(keypoints, res_file):
    """
    """

    results = []
    for img_kpts in keypoints:
        if len(img_kpts) == 0:
            continue

        _key_points = np.array([img_kpts[k]['keypoints']
                                for k in range(len(img_kpts))])
        key_points = np.zeros((_key_points.shape[0], 17 * 3), dtype=np.float)

        for ipt in range(17):
            key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
            key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
            key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

        result = [
            {
                'image_id': img_kpts[k]['image'],
                'category_id': 1,
                'keypoints': list(key_points[k]),
                'score': img_kpts[k]['score'],
                'center': list(img_kpts[k]['center']),
                'scale': list(img_kpts[k]['scale'])
            }
            for k in range(len(img_kpts))
        ]
        results.extend(result)

    return results


def convert_to_coco_format(pose_entries, all_keypoints):
    """
    Converting detected poses and keypoints list to the format accepted
    by COCO Evaluation

    Args:
    -----
    pose_entries: list
        list containing the pose entry information for all detected poses
    all_keypoints: list
        list with all indexed detected keypoints in the image

    Returns:
    --------
    coco_keypoints: list
        list with the detected keypoints in the format required by coco
    scores: list
        scores for the images
    """

    coco_keypoints = []
    scores = []

    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3  # 17 keypoints (x,y,visibility)
        person_score = pose_entries[n][-2]

        for position_id, keypoint_id in enumerate(pose_entries[n][:-3]):
            if SKIP_NECK == True and position_id == 1:  # no 'neck' keypoint in COCO => for lightweight open pose
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                visibility = 1
            keypoints[TO_COCO_MAP[position_id] * 3 + 0] = cx
            keypoints[TO_COCO_MAP[position_id] * 3 + 1] = cy
            keypoints[TO_COCO_MAP[position_id] * 3 + 2] = visibility

        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'

    # if no poses have been found, we add an empty pose
    if(len(pose_entries) == 0):
        keypoints = [0] * 17 * 3  # 17 keypoints (x,y,visibility)
        coco_keypoints.append(keypoints)
        scores.append(0)

    return coco_keypoints, scores


def convert_to_submission_dictionary(coco_keypoints, scores, img_name, coco_result=None):
    """
    Converting the list of detected keypoints into a dictionary for the coco eval submission

    Args:
    -----
    coco_keypoints: list
        list containing the coco keypoints with format (x,y,v)
    scores: list
        detection scores for each of the pafs
    img_name: string
        original name of the image. It is necessary since it has the img_code
    coco_result: list/None
        previous results already appended
    """

    if(coco_result is None):
        coco_result = []

    image_id = int(img_name.split('.')[0])
    for idx in range(len(coco_keypoints)):
        pts_add = []
        cur_points = coco_keypoints[idx]
        for i in range(len(cur_points)//3):
            y = cur_points[3*i + 0]
            x = cur_points[3*i + 1]
            v = cur_points[3*i + 2]
            pts_add += [x, y, v]

        coco_result.append({
            'image_id': image_id,
            'category_id': 1,  # person
            'keypoints': pts_add,
            'score': scores[idx]
        })

    return coco_result


def add_neck_keypoint(target):
    """
    Adding a new keypoint for the neck as the midpoint of the shoulders
    """

    for n in range(len(target)):
        keypoints = target[n]['keypoints']
        shoulder_1 = keypoints[5*3:(5+1)*3]
        shoulder_2 = keypoints[6*3:(6+1)*3]
        if(shoulder_1[2] == 0 and shoulder_2[2] == 0):
            vis = 0
        else:
            vis = 2
        keypoint = [(shoulder_1[0] + shoulder_2[0]) // 2, (shoulder_1[1] + shoulder_2[1]) // 2, vis]
        target[n]['keypoints'] = keypoints + keypoint

    return target


def generate_gaussian_kernel(kernel_size=15, sigma=10):
    """
    Pregenerating a gaussian kernel
    """

    x_axis = np.arange(0, 15) - 7
    y_axis = np.arange(0, 15) - 7
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(- (np.square(xx) + np.square(yy)) / np.square(sigma))

    global KERNEL
    KERNEL = kernel

    return


# @timimg
def generate_heatmaps(image, target, sigma=10):
    """
    Generating heatmaps by applying gaussian kernels to the joint keypoints

    Args:
    -----
    image: numpy array
        input image from the (styled) COCO dataset
    target: list/dictionary
        annotations loaded from the COCO annotation file
    sigma: integer
        standard deviation of the gaussian kernel

    Returns:
    --------
    keypoint_maps: numpy array
        3-d numpy array (number-keypoints+1, height, width) containing a heatmap for
        each of the keypoints and one (the last one) for a combination of all heatmaps
    """

    # obtaining number of joints and number of people from annotations
    n_people = len(target)
    if(n_people > 0):
        n_keypoints = len(target[0]['keypoints']) // 3
    else:
        n_keypoints = 17
    keypoint_maps = np.zeros((n_keypoints+1, image.shape[0], image.shape[1]), dtype=np.float32)

    # creating a heatmap for each joint for all people. Example: heatmap for left knee, ...
    for keypoint_idx in range(n_keypoints):
        aux_maps = np.zeros((n_people, image.shape[0], image.shape[1]), dtype=np.float32)
        for n in range(n_people):
            keypoint = target[n]['keypoints'][keypoint_idx*3:(keypoint_idx+1)*3]
            if keypoint[2] > 0:  # only adding gaussian kernel if point is visible
                aux_maps[n,:] = add_gaussian(aux_maps[n,:], keypoint[0], keypoint[1], sigma=sigma)
        keypoint_maps[keypoint_idx] = aux_maps.max(axis=0)

    # aggregated heatmaps for all joints and for all people
    keypoint_maps[-1,:] = 1 - keypoint_maps[:-1].max(axis=0)

    return keypoint_maps


def add_gaussian(keypoint_map, x, y, sigma=10):
    """
    Evaluation of a gaussian kernel centered at (y,x)

    Args:
    -----
    keypoint_map: numpy array
        array indicating the shape of the gaussian kernel
    x, y: integers
        center coordinates of the kernel
    sigma: integer
        standard deviation of the gaussian kernel

    Returns:
    --------
    keypoint_map: numpy array
        evaluation of the Gaussian kernel saved in the input keypoint_map
    """

    keypoint_map = np.zeros(keypoint_map.shape)

    # taking care of boundaries
    min_y = np.maximum(y-7, 0)
    off_min_y = abs(y-7 - min_y)
    max_y = np.minimum(y+8, keypoint_map.shape[0])
    off_max_y = 15-abs(y+8 - max_y)
    min_x = np.maximum(x-7, 0)
    off_min_x = abs(x-7 - min_x)
    max_x = np.minimum(x+8, keypoint_map.shape[1])
    off_max_x = 15-abs(x+8 - max_x)

    if(KERNEL is None):
        generate_gaussian_kernel()
    keypoint_map[min_y:max_y, min_x:max_x] = KERNEL[off_min_y:off_max_y, off_min_x:off_max_x]

    return keypoint_map


# @timimg
def generate_paf(image, target, thickness=5):
    """
    Generating the Part Affinity Fields (PAFs) for each limb

    Args:
    -----
    image: numpy array
        input image from the (styled) COCO dataset
    target: list/dictionary
        annotations loaded from the COCO annotation file
    thickness: integer
        thickness of the paf

    Returns:
    --------
    paf_maps: numpy array
        3-d numpy array (number-pafs + 1, height, width) containing a map for
        each of the libs and one (the last one) for a combination of all pafs
    """

    # variables for number of humans, limbs and keypoints
    n_pafs = len(BODY_PARTS_KPT_IDS)
    n_people = len(target)
    if(n_people > 0):
        n_keypoints = len(target[0]['keypoints']) // 3
    else:
        n_keypoints = 17
    paf_maps = np.zeros((n_pafs*2, image.shape[0], image.shape[1]), dtype=np.float32)

    # computing the paf for each limb
    for paf_idx in range(n_pafs):
        idx_a = BODY_PARTS_KPT_IDS[paf_idx][0]
        idx_b = BODY_PARTS_KPT_IDS[paf_idx][1]

        # computing for each person independently
        aux_maps_1 = np.zeros((n_people, image.shape[0], image.shape[1]), dtype=np.float32)
        aux_maps_2 = np.zeros((n_people, image.shape[0], image.shape[1]), dtype=np.float32)
        for n in range(n_people):

            if(idx_a >= n_keypoints or idx_b >= n_keypoints):
                continue
            keypoint_a = target[n]['keypoints'][idx_a*3:(idx_a+1)*3]
            keypoint_b = target[n]['keypoints'][idx_b*3:(idx_b+1)*3]

            if keypoint_a[2] > 0 and keypoint_b[2] > 0:
                aux_maps_1[n,:], aux_maps_2[n,:] = set_paf(image.shape[0], image.shape[1], keypoint_a,
                                                           keypoint_b, thickness=thickness)

        paf_maps[paf_idx*2,:] = np.mean(aux_maps_1, axis=0)
        paf_maps[paf_idx*2+1,:] = np.mean(aux_maps_2, axis=0)

    return paf_maps

# @timimg
def set_paf(height, width, a, b, thickness=5):
    """
    Computing the paf map given the keypoint coordinates

    Args:
    -----
    height, width: integers
        height and width of the image respectively
    x_a, y_a, x_b, y_b: integer
        x and y coordinates of the keypoints (a and b) between which the paf is formed
    thickness: integer
        thickness of the paf

    Returns:
    --------
    paf_map: numpy array
        matrix with the same shape as the image and with the paf set between the keypoints
    """

    a, b = np.array(a)[:2], np.array(b)[:2]
    a, b = a[::-1], b[::-1]
    paf_map_1 = np.zeros((height, width))
    paf_map_2 = np.zeros((height, width))

    y_ba = b[0] - a[0]
    x_ba = b[1] - a[1]
    x_min = int(max(min(b[1], a[1]) - thickness, 0))
    y_min = int(max(min(b[0], a[0]) - thickness, 0))
    x_max = int(min(max(b[1], a[1]) + thickness, width))
    y_max = int(min(max(b[0], a[0]) + thickness, height))

    norm_ba = (x_ba ** 2 + y_ba ** 2) ** 0.5
    if norm_ba < 1e-7:  # Same points, no paf
        return paf_map_1, paf_map_2
    x_ba = x_ba / norm_ba
    y_ba = y_ba / norm_ba

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(x, y)
    x_ca = xx - a[1]
    y_ca = yy - a[0]
    d = np.abs(x_ca * y_ba - y_ca * x_ba)

    idx = np.argwhere(d <= thickness)
    paf_map_1[idx[:,0] + y_min, idx[:,1] + x_min] = x_ba
    paf_map_2[idx[:,0] + y_min, idx[:,1] + x_min] = y_ba

    return paf_map_1, paf_map_2


# DEPRECATED: BRUTE FORCE APPROACH => TOO SLOW
def set_paf_(height, width, a, b, thickness=5):
    """
    Computing the paf map given the keypoint coordinates. Similar to the method above, but
    much less efficient. Currently not being used due to loss of efficiency.
    """

    a, b = np.array(a)[:2], np.array(b)[:2]
    a, b = a[::-1], b[::-1]
    paf_map_1 = np.zeros((height, width))
    paf_map_2 = np.zeros((height, width))
    col, row = np.ogrid[:height, :width]

    y_ba = b[0] - a[0]
    x_ba = b[1] - a[1]
    x_min = int(max(min(b[1], a[1]) - thickness, 0))
    y_min = int(max(min(b[0], a[0]) - thickness, 0))
    x_max = int(min(max(b[1], a[1]) + thickness, width))
    y_max = int(min(max(b[0], a[0]) + thickness, height))

    norm_ba = (x_ba ** 2 + y_ba ** 2) ** 0.5
    if norm_ba < 1e-7:  # Same points, no paf
        return
    x_ba = x_ba / norm_ba
    y_ba = y_ba / norm_ba

    # TODO: see how to make this a matrix operation
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            x_ca = x - a[1]
            y_ca = y - a[0]
            d = np.abs(x_ca * y_ba - y_ca * x_ba)
            if d <= thickness:
                paf_map_1[y, x] = x_ba
                paf_map_2[y, x] = y_ba

    return paf_map_1, paf_map_2


# @timimg
def get_mask(img, target):
    """
    Obtaining binary mask to only consider labeled humas

    Args:
    -----
    img: numpy array
        input image from the (styled) COCO dataset
    target: list/dictionary
        annotations loaded from the COCO annotation file

    Returns:
    --------
    mask: numpy array
        binary matrix segmenting only human labeled figurews
    """

    height, width = img.shape[0], img.shape[1]
    n_people = len(target)
    mask = np.ones((height, width))

    for n in range(n_people):
        cur_segmentation = target[n]["segmentation"]
        rle = pycocotools.mask.frPyObjects(cur_segmentation, height, width)
        if(len(pycocotools.mask.decode(rle).shape)==3):
            idx = (pycocotools.mask.decode(rle) > 0.5)[:,:,0]
        else:
            idx = (pycocotools.mask.decode(rle) > 0.5)
        mask[idx] = 0

    return mask


def upscale_heatmaps_pafs(heatmaps, pafs, target_size=400):
    """
    Upscaling heatmasp and pafs
    """

    # preprocessing
    heatmaps = F.interpolate(heatmaps.clone().detach(), target_size, mode="bilinear", align_corners=True)
    pafs = F.interpolate(pafs.clone().detach(), target_size, mode="bilinear", align_corners=True)

    return heatmaps, pafs


def rescale_predictions(prediction, stride=8):
    """
    Rescaling an input tensor given the stride
    """

    if(len(prediction.shape)==3):
        prediction = prediction[:, ::stride, ::stride]
    else:
        prediction = prediction[:, :, ::stride, ::stride]

    return prediction


def resize_inference(keypoints, original_size, cur_size=400):
    """
    Processing the keypoints so that they match the original shape of the image.
    Basically we invert the reshape transform
    """

    height, width = original_size[0], original_size[1]

    if(height > width):
        w = int(cur_size * width / height)
        pad_x = int((cur_size - w) / 2)
        pad_y = 0
    else:
        h = int(cur_size * height / width)
        pad_x = 0
        pad_y = int((cur_size - h) / 2)

    y_scale = height / (cur_size - pad_y * 2)
    x_scale = width / (cur_size - pad_x * 2)

    tf_keypoints = []
    for cur_keypoints in keypoints:
        cur_tf_keypoints = []
        for kpt_idx in range(len(cur_keypoints)//3):
            y = int(round((cur_keypoints[3 * kpt_idx + 0] - pad_y) * y_scale))
            x = int(round((cur_keypoints[3 * kpt_idx + 1] - pad_x) * x_scale))
            v = int(round(cur_keypoints[3 * kpt_idx + 2]))
            cur_tf_keypoints += [y, x, v]
        tf_keypoints.append(cur_tf_keypoints)

    return tf_keypoints

#
