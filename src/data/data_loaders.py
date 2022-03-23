"""
Methods for loading datasets and fitting data loaders for training and evaluation

@author: Angel Villar-Corrales
"""

import os

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from data import HRNetCoco, DetectionCoco, ArchDataset
from data.custom_transforms import ResizeImageDetection
from data.DirectoryDataset import DirectoryDataset
from CONFIG import CONFIG


def load_dataset(exp_data, train=True, validation=True, shuffle_train=False, shuffle_valid=False,
                 get_dataset=False, valid_size=0.2, perceptual_loss_dict=None, percentage=None):
    """
    Loading the dataset for human pose estimation and fitting data loaders to
    iterate the different splits

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment
    train, validation: boolean
        if True, a data loader is created for the given split
    shuffle_train, shuffle_train: boolean
        if True, images are accessed randomly
    get_dataset: boolean
        If True, dataset is fetched, otherwise only the dataloader
    valid_size: float
        Percentage [0, 1] of training data used for validation
    perceptual_loss_dict: TODO
        TODO
    percentage: TODO
        TODO
    """
    DATASETS = ["coco", "styled_coco", "arch_data", "combined"]

    data_path = CONFIG["paths"]["data_path"]
    batch_size = exp_data["training"]["batch_size"]
    dataset_name = exp_data["dataset"]["dataset_name"]
    if("alpha" not in exp_data["dataset"].keys()):
        exp_data["dataset"]["alpha"] = "0.5"
    alpha = exp_data["dataset"]["alpha"]
    if("styles" not in exp_data["dataset"].keys()):
        exp_data["dataset"]["styles"] = "redblack"
    styles = exp_data["dataset"]["styles"]
    labels_path = os.path.join(data_path, "annotations")

    train_loader, valid_loader = None, None
    train_set, valid_set = None, None
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Use one of {DATASETS}...")

    # loading training set
    if(train):
        if(dataset_name == "coco"):
            img_path = os.path.join(data_path, "original_images", "train2017")
            labels_file = os.path.join(labels_path, "person_keypoints_train.json")
            dataset = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=True,
                    is_styled=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        elif(dataset_name == "styled_coco"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "train")
            labels_file = os.path.join(labels_path, "person_keypoints_train.json")
            dataset = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=True,
                    is_styled=True,
                    alpha=alpha,
                    styles=styles,
                    perceptual_loss_dict=perceptual_loss_dict,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        elif(dataset_name == "arch_data"):
            dataset = ArchDataset(
                    task="pose_estimation",
                    split_set="train",
                    shuffle=shuffle_train,
                    valid_size=valid_size,
                    percentage=percentage,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize
                    ])
                )
        elif(dataset_name == "combined"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "train")
            labels_file = os.path.join(labels_path, "person_keypoints_train.json")
            dataset1 = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=True,
                    is_styled=True,
                    alpha=alpha,
                    styles=styles,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            dataset2 = ArchDataset(
                    task="pose_estimation",
                    split_set="train",
                    shuffle=shuffle_train,
                    valid_size=valid_size,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize
                    ])
                )
            dataset = ConcatDataset(dataset1, dataset2)

        train_loader = get_dataset_loader(dataset, batch_size=batch_size, shuffle=shuffle_train)
        train_set = dataset

    # Loading validation set
    if(validation):
        if(dataset_name == "coco"):
            img_path = os.path.join(data_path, "original_images", "val2017")
            labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
            dataset = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=False,
                    is_styled=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        elif(dataset_name == "styled_coco"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "validation")
            labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
            dataset = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=False,
                    is_styled=True,
                    alpha=alpha,
                    styles=styles,
                    perceptual_loss_dict=perceptual_loss_dict,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        elif(dataset_name == "arch_data"):
            dataset = ArchDataset(
                    task="pose_estimation",
                    split_set="validation",
                    shuffle=shuffle_valid,
                    valid_size=valid_size,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
        elif(dataset_name == "combined"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "validation")
            labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
            dataset1 = HRNetCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=False,
                    is_styled=True,
                    alpha=alpha,
                    styles=styles,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            dataset2 = ArchDataset(
                    task="pose_estimation",
                    split_set="validation",
                    shuffle=shuffle_valid,
                    valid_size=valid_size,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            dataset = ConcatDataset(dataset2, dataset1)

        valid_loader = get_dataset_loader(dataset, batch_size=batch_size, shuffle=shuffle_valid)
        valid_set = dataset

    if(get_dataset):
        return train_loader, valid_loader, train_set, valid_set
    else:
        return train_loader, valid_loader


def get_detection_dataset(exp_data, train=True, validation=True, shuffle_train=False,
                          shuffle_valid=False, get_dataset=False, percentage=None,
                          class_ids=[1], perceptual_loss_dict=None, valid_size=0.2):
    """
    Loading the detection dataset and fitting data loaders to iterate the different splits

    Args:
    -----
    exp_data: dictionary
        parameters corresponding to the different experiment
    train, validation: boolean
        if True, a data loader is created for the given split
    shuffle_train, shuffle_train: boolean
        if True, images are accessed randomly
    class_ids: list of integers
        list containing the ids of the classes to detect. By defaul [1] (person class)
    """
    DATASETS = ["coco", "styled_coco", "arch_data", "combined", "red_black", "open_subset"]

    data_path = CONFIG["paths"]["data_path"]
    batch_size = exp_data["training"]["batch_size"]
    dataset_name = exp_data["dataset"]["dataset_name"]
    if("alpha" not in exp_data["dataset"].keys()):
        exp_data["dataset"]["alpha"] = "0.5"
    alpha = exp_data["dataset"]["alpha"]
    if("styles" not in exp_data["dataset"].keys()):
        exp_data["dataset"]["styles"] = "redblack"
    styles = exp_data["dataset"]["styles"]
    labels_path = os.path.join(data_path, "annotations")

    train_loader, valid_loader = None, None
    train_set, valid_set = None, None

    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported. Use one of {DATASETS}...")

    # Loading training set
    if(train):
        if(dataset_name == "coco"):
            img_path = os.path.join(data_path, "original_images", "train2017")
            labels_file = os.path.join(labels_path, "person_keypoints_train.json")
            dataset = DetectionCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=True,
                    is_styled=False,
                    class_ids=class_ids,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    perceptual_loss_dict=None,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )
        elif(dataset_name == "styled_coco"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "train")
            labels_file = os.path.join(labels_path, "person_keypoints_train.json")
            dataset = DetectionCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=True,
                    is_styled=True,
                    class_ids=class_ids,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    perceptual_loss_dict=perceptual_loss_dict,
                    alpha=alpha, styles=styles,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )
        elif(dataset_name == "arch_data"):
            dataset = ArchDataset(
                    task="person_detection",
                    shuffle=shuffle_train,
                    split_set="train",
                    valid_size=valid_size,
                    percentage=percentage,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )

        train_loader = get_dataset_loader(dataset, batch_size, shuffle_train)
        train_set = dataset

    # Loading training set
    if(validation):
        if(dataset_name == "coco"):
            img_path = os.path.join(data_path, "original_images", "val2017")
            labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
            dataset = DetectionCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=False,
                    is_styled=False,
                    class_ids=class_ids,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    perceptual_loss_dict=None,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )
        elif(dataset_name == "styled_coco"):
            img_path = os.path.join(data_path, f"images_style_{styles}_alpha_{alpha}", "validation")
            labels_file = os.path.join(labels_path, "person_keypoints_validation.json")
            dataset = DetectionCoco(
                    exp_data=exp_data,
                    root=data_path,
                    img_path=img_path,
                    labels_path=labels_file,
                    is_train=False,
                    is_styled=True,
                    class_ids=class_ids,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    perceptual_loss_dict=perceptual_loss_dict,
                    alpha=alpha, styles=styles,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )
        elif(dataset_name == "arch_data"):
            dataset = ArchDataset(
                    task="person_detection",
                    split_set="validation",
                    shuffle=shuffle_valid,
                    valid_size=valid_size,
                    resizer=ResizeImageDetection(exp_data["dataset"]["image_size"]),
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])
                )
        elif(dataset_name == "red_black"):
            datapath = os.path.join(data_path, "class_arch_red_black")
            dataset = DirectoryDataset(datapath=datapath)
        elif(dataset_name == "open_subset"):
            datapath = os.path.join(data_path, "class_arch_open_access", "images")
            dataset = DirectoryDataset(datapath=datapath)

        valid_loader = get_dataset_loader(dataset, batch_size, shuffle_valid, collate=True)
        valid_set = dataset

    if(get_dataset):
        return train_loader, valid_loader, train_set, valid_set
    else:
        return train_loader, valid_loader


def collate_fn(batch):
    """ Custom collator """
    return tuple(zip(*batch))


def get_dataset_loader(dataset, batch_size=64, shuffle=False, collate=None):
    """
    Fitting a dataset split into a data loader

    Args:
    -----
    dataset: string
        name of the dataset to load
    batch_size: integer
        number of elements in each batch
    shuffle: boolean
        if True, images are accessed randomly
    collate: function
        Argument for providing a custom collator

    Returns:
    --------
    data_loader: DataLoader
        data loader to iterate the dataset split using batches
    """
    data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )
    if collate is not None:
        data_loader.collate_fn = collate

    return data_loader


def get_vase_subset(img_size=None):
    """
    Loading a small subset of vase painting data for qualitative evaluation

    Returns:
    --------
    imgs: list
        list containing subset images as a torch Tensor with shape [(3,H,W)]
    """

    # reading image names from the data directory
    data_path = CONFIG["paths"]["data_path"]
    data_path = os.path.join(data_path, "ccoimages_final")
    img_names = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    resizer = ResizeImageDetection(img_size)

    # loading images and converting to desired shape and data type
    imgs = []
    for img_name in sorted(img_names):
        if(".jpg" not in img_name):
            continue
        data_numpy = cv2.imread(
            img_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if(img_size is not None):
            data_numpy = resizer(data_numpy)
        data_numpy = data_numpy.transpose(2, 0, 1)
        imgs.append(torch.from_numpy(data_numpy))
    return imgs


class ConcatDataset(Dataset):
    """
    Dataset object used to combine two different datasets (e.g, for 'Combined')

    Args:
    -----
    datasets: iterable or *args
        Different datasets to concatenate
    """

    def __init__(self, *datasets):
        """ Module initializer """
        self.datasets = datasets
        self.length = np.sum([len(d) for d in datasets])
        db_lims = []
        cur_lim = 0
        for d in datasets:
            db_lims.append(cur_lim + len(d))
            cur_lim = cur_lim + len(d)
        self.db_lims = np.array(db_lims)
        return

    def __getitem__(self, i):
        """ Finding to which dataset the sample belongs and sampling """
        db_idx = np.where(self.db_lims > i)[0][0]
        cur_dataset = self.datasets[db_idx]
        sample_idx = i if db_idx == 0 else i - self.db_lims[db_idx - 1]
        return cur_dataset[sample_idx]

    def __len__(self):
        """ Total number of elements. Sum of lengths of individual datasets """
        return self.length

#
