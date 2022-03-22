"""
Methods for visualization of images, targets and overlays

EnhancePoseEstimation/src/lib
@author: 
"""

import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import torch
import lib.pose_parsing as pose_parsing
import lib.transforms as custom_transforms

COLORS = {
    "0": "red",             # left ankle -  left knee
    "1": "tomato",          # left knee  -  left hip
    "2": "lime",            # left hip   - left shoulder
    "3": "royalblue",       # right hip  - right knee
    "4": "navy",            # right knee - right ankle
    "5": "green",           # right hip  - right shoulder
    "6": "bisque",
    "7": "palegoldenrod",
    "8": "khaki",
    "9": "moccasin",
    "10": "wheat",
    "11": "fuchsia",        # left elbow     - left wrist
    "12": "deeppink",       # left shouldsr  - left elbow
    "13": "lawngreen",      # left shoulder  - right shoulder
    "14": "aqua",           # right shoulder - right elbow
    "15": "turquoise",      # right elbow    - right wrist
    "16": "darkorange",     # left shoulder  - left ear
    "17": "orange",         # right shoudler - right ear
    "18": "saddlebrown"
}


def draw_pose(img, poses, all_keypoints, **kwargs):
    """
    Overlaying the predicted poses on top of the images

    Args:
    -----
    img: numpy array
        Array corresponding to the image to overlay the pose on. By default: (H,W,3)
    poses: numpy array
        Array with the keypoint indices for each of the poses.
        Example:
        --------
        poses = [[0,1,2,...,17,17,1], [18,19,....,33,17,1]]
            => There are two poses. The first one uses keypoints 0-17 and the
               second one 18-33
    all_keypoints: numpy array
        Array with all keypoints (x,y,v). Shape is (N_keypoints, 3).
        If v coordinate is 0, point is not drawn.
    """

    if("ax" not in kwargs):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs and kwargs["preprocess"] == True):
        img = custom_transforms.unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)
    ax.imshow(img)

    for pose in poses:
        pose = pose[:-2]  # removing num_kpts and score
        for idx, limb in enumerate(pose_parsing.SKELETON):
            # drawing circles
            if(limb[0] < 0 and limb[1] < 0): # for simplified skeleton
                continue
            idx_a, idx_b = int(pose[limb[0]]), int(pose[limb[1]])
            if(idx_a == -1 or idx_b == -1):
                continue
            a, b = all_keypoints[idx_a], all_keypoints[idx_b]
            if(a[-1] == 0 or b[-1] == 0):
                continue
            if( (a[0]+a[1])<=1 or (b[0]+b[1])<=1):
                continue
            color = COLORS[str(idx)]
            line = mlines.Line2D(
                    np.array([a[1], b[1]]), np.array([a[0], b[0]]),
                    ls='-', lw=5, alpha=1, color=color)
            circle1 = mpatches.Circle(np.array([a[1], a[0]]), radius=5,
                                     ec='black', fc=color,
                                     alpha=1, linewidth=5)
            circle2 = mpatches.Circle(np.array([b[1], b[0]]), radius=5,
                                     ec='black', fc=color,
                                     alpha=1, linewidth=5)
            line.set_zorder(1)
            circle1.set_zorder(2)
            circle2.set_zorder(2)
            if(limb[0] >= 0):
                ax.add_patch(circle1)
            if(limb[1] >= 0):
                ax.add_patch(circle2)
            if(limb[0] >= 0 and limb[1] >= 0):
                ax.add_line(line)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])
    if("axis_off" in kwargs):
        ax.axis("off")
    if("savepath" in kwargs):
        plt.savefig(kwargs["savepath"],
                    bbox_inches='tight',
                    pad_inches=0)

    return ax


def draw_skeleton(kpts, shape=(256,192,3), **kwargs):
    """
    Drawing a skeleton on a blank image
    """

    blank_image = np.zeros(shape)
    kpts[:,-1] = 1  # adding visibility
    kpts = [kpts[:, 1], kpts[:, 0], kpts[:, 2]]
    kpts = np.array(kpts).T
    draw_pose(img=blank_image, poses=[np.arange(19)], all_keypoints=kpts, **kwargs)

    return


def visualize_image(img, **kwargs):
    """
    Visualizing an image accouting for the BGR format
    """

    if("ax" not in kwargs):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
    else:
        ax = kwargs["ax"]
    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs):
        img = custom_transforms.unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)

    ax.imshow(img)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])
    if("axis_off" in kwargs):
        ax.axis("off")
    if("savepath" in kwargs):
        plt.savefig(kwargs["savepath"],
                    bbox_inches='tight',
                    pad_inches=0)

    return


def visualize_instances(imgs, **kwargs):
    """
    Displaying all person instances in the image in one row
    """

    n_humans = imgs.shape[0]
    fig, ax = plt.subplots(1,n_humans)
    fig.set_size_inches(5*n_humans, 5)
    for i in range(n_humans):
        if(n_humans == 1):
            visualize_image(np.array(imgs[i,:]), ax=ax, title=f"Person {i+1}",
                            preprocess=True)
        else:
            visualize_image(np.array(imgs[i,:]), ax=ax[i], title=f"Person {i+1}",
                            preprocess=True)
    plt.tight_layout()
    plt.show()

    return


def overlay_heatmap(heatmap, img, ax=None, **kwargs):
    """
    Overlaying a heatmap on top of the original image
    """

    # heatmap_vis = (heatmap) * 255
    # heatmap_vis = np.round(heatmap_vis).astype(int)

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    img = custom_transforms.unnormalize(torch.Tensor(img))
    img = img.numpy().transpose(1,2,0)

    disp = img
    disp[:,:,0] = img[:,:,0] + heatmap
    disp = np.clip(disp, a_min=0, a_max=1)
    ax.imshow(disp)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])
    if("axis_off" in kwargs):
        ax.axis("off")

    return


def overlay_paf(pafs, img, ax=None, **kwargs):
    """
    Overlaying pafs on top of the original image
    """

    pafs[np.abs(pafs)<1e-1] = 0
    paf_disp = np.logical_or(pafs[0,:], pafs[1, :])*255

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    image_vis = img.transpose(1,2,0)*256 + 128
    image_vis = np.round(image_vis).astype(int)

    disp = image_vis
    disp[:,:,0] = paf_disp
    disp = np.round(disp).astype(int)
    ax.imshow(disp)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    return


def visualize_heatmap(heatmap, ax, fig=None, **kwargs):
    """
    Visualizing the heatmaps in the given figure
    """

    heatmap_vis = (1 - heatmap)*255
    heatmap_vis = np.round(heatmap_vis).astype(int)
    ax.imshow(heatmap_vis)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    return


def visualize_paf(paf, ax, fig=None, **kwargs):
    """
    Visualizing the paf in the given figure
    """

    img_vis = np.zeros(paf.shape)
    idx = np.argwhere(paf>0)
    img_vis[idx[:,0],idx[:,1]] = 255

    ax.imshow(img_vis)

    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    return


def visualize_bbox(img, boxes, labels=None, scores=None, ax=None, **kwargs):
    """
    Visualizing the bounding boxes and scores predicted by the faster rcnn model

    Args:
    -----
    img: numpy array
        RGB image that has been predicted
    boxes: numpy array
        Array of shape (N, 4) where N is the number of boxes detected.
        The 4 corresponds to y_min, x_min, y_max, x_max
    labels: numpy array
        Array containing the ID for the predicted labels
    scores: numpy array
        Array containing the prediction confident scores
    """

    # initializing axis object if necessary
    if(ax is None):
        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(6,6)

    if("bgr" in kwargs and kwargs["bgr"] == True):
        img = np.array([img[2,:,:], img[1,:,:], img[0,:,:]])
    if("preprocess" in kwargs and kwargs["preprocess"] == True):
        img = custom_transforms.unnormalize(torch.Tensor(img))
        img = img.numpy().transpose(1,2,0)
    ax.imshow(img)


    if("title" in kwargs):
        ax.set_title(kwargs["title"])

    # in case of no detections
    if len(boxes) == 0:
        if("savepath" in kwargs):
            plt.axis("off")
            plt.savefig(kwargs["savepath"], bbox_inches='tight', pad_inches=0)
        return

    # fetching ArchData labels if available
    arch_labels = None if("arch_labels" not in kwargs) else kwargs["arch_labels"]
    # displaying BBs
    for i, bb in enumerate(boxes):
        x, y = bb[0], bb[1]
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        ax.add_patch(plt.Rectangle((x, y), width, height, fill=False,
                                    edgecolor='red', linewidth=5))

        message = None
        if(scores is not None or arch_labels is not None):
            cur_score = f": {round(float(scores[i]),2)}" if scores is not None else ""
            cur_arch_lbl = arch_labels[i] if arch_labels is not None else "Score"
            message = f"{cur_arch_lbl}{cur_score}"
        if(message is not None):
            ax.text(bb[0], bb[1], message, style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    if("axis_off" in kwargs and kwargs["axis_off"]==True):
        plt.axis("off")
    if("savepath" in kwargs):
        plt.savefig(kwargs["savepath"], bbox_inches='tight', pad_inches=0)
        # plt.savefig(kwargs["savepath"])

    return


def visualize_subset_heatmaps(img, heatmaps, savepath=None, fig=None):
    """
    Visualizing a subset of images, the corresponding heatmaps and an overlay
    """

    if(savepath is not None):
        plt.figure()

    for i in range(3):

        cur_img = np.array(img[i,:])
        cur_img = np.array([cur_img[2,:,:], cur_img[1,:,:], cur_img[0,:,:]])
        cur_img = cur_img.transpose(1,2,0)*256 + 128
        cur_img = np.round(cur_img).astype(int)
        cur_heatmap = np.array(heatmaps[i,-1,:])

        plt.subplot(3,3,3*i+1)
        plt.imshow(cur_img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(3,3,3*i+2)
        heatmap_vis = (1 - cur_heatmap)*255
        heatmap_vis = np.round(heatmap_vis).astype(int)
        plt.imshow(heatmap_vis)
        plt.title(f"Keypoint Maps")
        plt.axis("off")

        plt.subplot(3,3,3*i+3)
        img_vis = cur_img[:,:,:]*0.5
        img_vis[:,:,0] = img_vis[:,:,0] + heatmap_vis
        img_vis[:,:,1] = img_vis[:,:,1] + heatmap_vis
        img_vis[:,:,2] = img_vis[:,:,2] + heatmap_vis
        img_vis = np.round(img_vis).astype(int)
        plt.imshow(img_vis)
        plt.title(f"Overlay")
        plt.axis("off")
    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    return


def visualize_subset_pafs(img, pafs, savepath=None, fig=None):
    """
    Visualizing a subset of images, the corresponding heatmaps and an overlay
    """

    if(savepath is not None):
        plt.figure()

    for i in range(3):
        cur_img = np.array(img[i,:])
        cur_img = np.array([cur_img[2,:,:], cur_img[1,:,:], cur_img[0,:,:]])
        cur_img = cur_img.transpose(1,2,0)*256 + 128
        cur_img = np.round(cur_img).astype(int)
        cur_paf = np.abs(pafs[i,0,:]) + np.abs(pafs[i,4,:]) + np.abs(pafs[i,8,:])

        plt.subplot(3,3,3*i+1)
        plt.imshow(cur_img)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(3,3,3*i+2)
        cur_paf_vis = np.round(np.clip(cur_paf*1000, 0, 255)).astype(int)
        plt.imshow(cur_paf_vis)
        plt.title(f"PAFs")
        plt.axis("off")

        plt.subplot(3,3,3*i+3)
        idx = np.argwhere(cur_paf>0)
        cur_img[idx[:,0],idx[:,1],0] = 255
        cur_img[idx[:,0],idx[:,1],1] = 0
        cur_img[idx[:,0],idx[:,1],2] = 0
        cur_img = np.clip(cur_img, 0, 255)
        cur_img = np.round(cur_img).astype(int)
        plt.imshow(cur_img)
        plt.title(f"Overlay")
        plt.axis("off")
    plt.tight_layout()

    if(savepath is not None):
        plt.savefig(savepath)

    return
