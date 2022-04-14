"""
Transforms used during the preprocessing steps: normalizing to zero-mean and unit variance,
flipping image and keypoints, affine transforms for crops and bounding boxes, ...

@author: Angel Villar-Corrales
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


class TransformDetection():
    """
    Extracting a detection from an image given the bbox coordiantes. The detection
    keeps a certain aspect ration and it should be centered at the person
    """

    def __init__(self, det_width=192, det_height=256):
        """ Initializer of the Transform object """
        self.det_width = det_width
        self.det_height = det_height
        self.image_size = np.array([det_width, det_height])
        self.aspect_ratio = self.det_width * 1.0 / self.det_height
        self.pixel_std = 200

        return

    def __call__(self, img, list_coords):
        """ Extracting the detection given the full image and the bbox coords """
        detections = []
        centers = []
        scales = []

        for coords in list_coords:
            center, scale = self._coords2cs(coords)
            trans = get_affine_transform(center=center, scale=scale,
                                         rot=0, output_size=self.image_size)
            detection = cv2.warpAffine(
                            img,
                            trans,
                            (int(self.image_size[0]), int(self.image_size[1])),
                            flags=cv2.INTER_LINEAR
                        )
            detections.append(detection)
            centers.append(center)
            scales.append(scale)

        detections = np.array(detections)
        centers = np.array(centers)
        scales = np.array(scales)
        if(len(detections) == 0):
            return detections, centers, scales
        try:
            detections = detections.transpose(0, 3, 1, 2)
        except ValueError:
            print(detections.shape)
            exit()
        return detections, centers, scales

    def _coords2cs(self, coords):
        """ Converting a bounding box (xmin, ymin, xmax, ymax) to (center, scale) """
        xmin, ymin, xmax, ymax = coords
        x, y = xmin, ymin
        w, h = (xmax - xmin), (ymax - ymin)

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


class Normalize():
    """
    Normalizing images by substracting mean and dividing by standard deviation.
    By defualt, images are standarized to be in the range [-1,1]

    Args:
    -----
    mean: tuple
        tuple containing the mean for each of the color channels. Default = (128, 128, 128)
    std: int
        standard deviation value. Default = 256
    """

    def __init__(self, mean=(128, 128, 128), std=256):
        """ Initializer of the Normalizer object """

        self.mean = mean
        self.std = std

        return

    def __call__(self, img):
        """ Normalizing the image """

        img = np.array(img, dtype=np.float32)
        norm_img = (img - self.mean) / self.std

        return norm_img


def prepare_image_detector(img):
    """
    Preprocessing an image to be fed into the faster rcnn

    Args:
    -----
    img: numpy array
        Image as an ndarry with shape (H,W,3) and in range [0,255]

    Returns:
    --------
    img: torch Tensor
        Image as an torch Tensor with shape (1,3,H,W) and in range [0,1]
    """
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, *img.shape)
    img = torch.Tensor(img) / 255
    return img


def unnormalize(img):
    """ Undoing the default COCO normalization. Output image will be in range [0,1] """
    if(torch.max(img) > 50):
        img = img / 255
        return img
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    unnormalize_img = transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())
    img = unnormalize_img(img)
    return img


def flip_back(output_flipped, matched_parts):
    """
    ouput_flipped: numpy array
        (batch_size, num_joints, height, width)
    """
    assert output_flipped.ndim == 4, 'output_flipped should be [batch_size, num_joints, height, width]'

    if(torch.is_tensor(output_flipped)):
        output_flipped = output_flipped.cpu().detach().numpy()

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return torch.from_numpy(output_flipped.copy())


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    Fliping coords of the joins for the flipped-image evaluating
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    """
    Transforming some coordinates using affine transformations
    """

    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        # target_coords[p, 0], target_coords[p, 1] = target_coords[p, 1], target_coords[p, 0]
    return target_coords


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    Obtaining the projection matrix that applies an affine transform from the original
    source points to the target points. If inv==1, then the invese matrix is returned
    """

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(f"WARNING! {type(scale)}")
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """ Applying an affine trainsform to an image or image section """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    """ """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """ Rotating a keypoint coordinate by a certain amount of angles """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def crop(img, center, scale, output_size, rot=0):
    """ Cropping and scaling a part of an image given its center and scale """
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


#
