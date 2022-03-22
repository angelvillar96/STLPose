"""
Constansts (e.g., keypoint order and paf connections) for the different pretrained models and
for the inference logic

EnhancePoseEstimation/src
@author: 
"""

import numpy as np

# Mapping keypoint index to semantic meaning of the part
IDX_TO_KPT_NAME_OPEN_POSE = {0: 'Nose', 1: 'Neck', 2: 'Right Shoulder', 3: 'Right Elbow',
                             4: 'Right Wrist', 5: 'Left Shoulder', 6: 'Left Elbow',
                             7: 'Left Wrist', 8: 'Right Hip', 9: 'Right Knee', 10: 'Right Ankle',
                             11: 'Left Hip', 12: 'Left Knee', 13: 'Left Ankle', 14: 'Right Eye',
                             15: 'Left Eye', 16: 'Right Ear', 17: 'Left Ear'}

IDX_TO_KPT_NAME = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
                   5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
                   9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
                   13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'}

KPT_NAME_TO_IDX = {'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
                   'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
                   'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
                   'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16}

IDX_TO_KPT_NAME_ARCH_DATA = {0:'Head', 1:'Neck', 2:'Thorax', 3:'Pelvis', 4:'Right Shoulder',
                             5:'Right Elbow', 6:'Right Wrist', 7:'Right Hip', 8:'Right Knee',
                             9:'Right Ankle', 10:'Right Toe', 11:'Left Shoulder',
                             12:'Left Elbow', 13:'Left Wrist', 14:'Left Hip', 15:'Left Knee',
                             16:'Left Ankle', 17:'Left Toe'}
KPT_NAME_TO_IDX_ARCH_DATA = {'Head':0, 'Neck':1, 'Thorax':2, 'Pelvis':3, 'Right Shoulder':4,
                             'Right Elbow':5, 'Right Wrist':6, 'Right Hip':7, 'Right Knee':8,
                             'Right Ankle':9, 'Right Toe':10, 'Left Shoulder':11,
                             'Left Elbow':12, 'Left Wrist':13, 'Left Hip':14, 'Left Knee':15,
                             'Left Ankle':16, 'Left Toe':17}
ARCHDATA_LBLS_TO_COCO = {
    "Head Top / Forehead": "Head", "Upper Neck": "Neck", "Pelvis": "Pelvis", 'Thorax':'Thorax',
    "Right Shoulder": "Right Shoulder", "Right Elbow":"Right Elbow",
    "Right Wrist": "Right Wrist", 'Right Hip':'Right Hip', 'Right Knee':'Right Knee',
    'Right Ankle':'Right Ankle', 'Right Toe':'Right Toe', 'Left Shoulder':'Left Shoulder',
    'Left Elbow':'Left Elbow', 'Left Wrist':'Left Wrist', 'Left Hip':'Left Hip',
    'Left Knee':'Left Knee', 'Left Ankle':'Left Ankle', 'Left Toe':'Left Toe'
}


# Part pairs for the rendering of the PAFs
PAFS_OPEN_POSE = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                  [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]

# map for reordering the keypoints for the lightweight OpenPose model
REORDER_MAP = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
REORDER_MAP_ARCH_DATA = [0, -1, -1, -1, -1, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 9]

# maps for grouping points and rendering the skeleton
SKELETON_OPEN_POSE = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                      [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
                      [2, 16], [5, 17]]

SKELETON_HRNET = [[15, 13], [13, 11], [11, 5], [12, 14], [14, 16], [12, 6], [3, 1], [1, 2],
                  [1, 0], [0, 2], [2, 4], [9, 7], [7, 5], [5, 6], [6, 8], [8, 10], [3, 5],
                  [4, 6]]

SKELETON_SIMPLE = [[15, 13], [13, 11], [11, 5], [12, 14], [14, 16], [12, 6], [-3, -1],
                   [-1, -2], [-1, 0], [0, -2], [-2, -4], [9, 7], [7, 5], [5, 6], [6, 8],
                   [8, 10], [0, 5], [0, 6]]  # connections to eyes and ears removed (-id)

SKELETON_ARCH_DATA = [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6], [1,11], [11,12], [12,13],
                      [3,7], [7,8], [8,9], [9,10], [3,14], [14,15], [15,16], [16,17]]


# also necessary fro joinig stuff
BODY_PARTS_PAF_IDS_OPEN_POSE = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1],
                                [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29], [30, 31],
                                [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])

# maps for converting to evaluation format
COCO_MAP_OPEN_POSE = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
COCO_MAP_HRNET = np.arange(17)

# label pairs to be flipped during mirroring
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8],
              [9, 10], [11, 12], [13, 14], [15, 16]]


def setup_body_part_paf_id_maps(model_name):
    """
    Selecting skeletion map assignment given the name of the model.
    Since different models use different pafs, the assignment map is different
    """

    if("OpenPose"in model_name):
        return BODY_PARTS_PAF_IDS_OPEN_POSE

    elif(model_name == "HRNet"):
        return BODY_PARTS_PAF_IDS_OPEN_POSE
        raise NotImplementedError("Logic for HRNet is not yet implemented")

    assert False
    return None


def setup_skeleton_map(model_name):
    """
    Selecting skeletion map assignment given the name of the model.
    Since different models use different pafs, the assignment map is different
    """

    if("OpenPose"in model_name):
        return SKELETON_OPEN_POSE

    elif(model_name == "HRNet"):
        return SKELETON_OPEN_POSE
        # raise NotImplementedError("Logic for HRNet is not yet implemented")

    assert False
    return None


def setup_pafs_maps(model_name):
    """
    Selecting paf map assignment given the name of the model.
    Different pretrained model expect different parts and create different pafs
    """

    if("OpenPose"in model_name):
        return PAFS_OPEN_POSE

    elif(model_name == "HRNet"):
        return PAFS_OPEN_POSE
        raise NotImplementedError("Logic for HRNet is not yet implemented")

    assert False
    return None


def setup_submission_maps(model_name):
    """
    Selecting the maps to reorder the keypoints for the submission format
    """

    if("OpenPose"in model_name):
        return COCO_MAP_OPEN_POSE, True

    elif(model_name == "HRNet"):
        return COCO_MAP_HRNET, False
        raise NotImplementedError("Logic for HRNet is not yet implemented")

    assert False
    return None

#
