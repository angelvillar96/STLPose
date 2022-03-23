"""
Classes and methods for transforming and augmenting the data

@author: Angel Villar-Corrales
"""

import numpy as np
import cv2
from skimage import transform as sktsf


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

    def __call__(self, img):
        """ Normalizing the image """
        norm_img = (img - self.mean) / self.std
        return norm_img


class ResizeImageDetection():
    """
    Converting image and bbox coords to a fixed image size.
    Image is downscaled so that largest side is converted to desired size,
    smalles side is padded with zeros
    """

    def __init__(self, img_size=400):
        """ Module initilizer """
        self.img_size = img_size

    def __call__(self, image, annots=None):
        """ Resizing image to the desired size while keeping the desired ratio"""
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # resizing and padding
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots["boxes"][:, :4] *= scale

        if(annots is None):
            return image
        return new_image, annots, scale


class Resize():
    """
    Class for Scaling the input images to a certain size while
    keeping the original image aspect ratio
    """

    def __init__(self, size=400):
        """ Intializer of the scaler """
        self.size = size

    def __call__(self, data):
        """ Scaling the received image and target to the desired shape

        Args:
        -----
        data: dictionary
            dictionary containing the image, heatmaps, pafs and mask

        Returns:
        --------
        resized_data: dictionary
            dictionary similar to the input, but with the resized data
        """

        if(isinstance(data, np.ndarray)):
            image = data
            heatmaps, pafs, mask = None, None, None
        else:
            image, heatmaps, pafs, mask = data["img"], data["heatmaps"], data["pafs"], data["mask"]
        height, width = image.shape[0], image.shape[1]

        if(height > width):
            w = int(self.size*width/height)
            h = self.size
            pad_val = int((self.size-w)/2)
            pad_idx = (0, h, pad_val, pad_val+w)
        else:
            h = int(self.size*height/width)
            w = self.size
            pad_val = int((self.size-h)/2)
            pad_idx = (pad_val, pad_val+h, 0, w)

        resized_img = np.zeros((self.size, self.size, 3))
        dim = (w, h)
        resized_img[pad_idx[0]:pad_idx[1], pad_idx[2]:pad_idx[3]] = \
            cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if(mask is not None):
            resized_mask = np.ones((1, self.size, self.size))
            resized_mask[:, pad_idx[0]:pad_idx[1], pad_idx[2]:pad_idx[3]] = \
                cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

        resized_pafs = []
        if(pafs is not None):
            for paf in pafs:
                cur_paf = np.zeros((self.size, self.size))
                cur_paf[pad_idx[0]:pad_idx[1], pad_idx[2]:pad_idx[3]] = \
                    cv2.resize(paf, dim, interpolation=cv2.INTER_AREA)
                resized_pafs.append(cur_paf)

        resized_heatmaps = []
        if(heatmaps is not None):
            for heatmap in heatmaps:
                cur_heatmap = np.zeros((self.size, self.size))
                cur_heatmap[pad_idx[0]:pad_idx[1], pad_idx[2]:pad_idx[3]] = \
                    cv2.resize(heatmap, dim, interpolation=cv2.INTER_AREA)
                resized_heatmaps.append(cur_heatmap)

        if(isinstance(data, np.ndarray)):
            resized_data = resized_img
        else:
            resized_data = {
                "img": resized_img,
                "heatmaps": np.array(resized_heatmaps),
                "pafs": np.array(resized_pafs),
                "mask": resized_mask
            }

        return resized_data


def preprocess_fast_rcnn(img, min_size=600, max_size=1000):
    """
    Preprocessing an image for feature extraction.
    The length of the shorter edge is scaled to 'min_size'.
    After resizing the image, the image is subtracted by a mean image value.

    Args:
    -----
    img: numpy.array
        An image. This is in CHW and RGB format and in range [0, 255].
    min_size, max_size: integer
        The image is scaled so that the smaller edge is at least 'min_size' and the
        longer edge is at least 'max_size'

    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    norm_img = (img - mean).astype(np.float32, copy=True)

    return norm_img

#
