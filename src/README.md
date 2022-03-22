# Enhancing Human Pose Estimation in Ancient Vase Paintings via Perceptually-grounded Style Transfer Learning

In this README, we further explain the different scripts and functionalities of the repository.

## Contents

 * [1. Creating a New Experiment](#creating-a-new-experiment)
 * [2. Person Detection](#person-detection)
 * [3. Pose Estimation](#pose-estimation)


## Creating a New Experiment

Creating an experiment automatically generates a directory in the specified experiments/EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and subdirectories for the models and plots.

Running the following, we display the complete list of parameters used in the experiments:

```shell
$ python 01_create_experiment.py --help
  usage: 01_create_experiment.py [-h --help]
                               [-d --exp_directory EXP_DIRECTORY required]
                               [--dataset_name DATASET_NAME required]
                               [--image_size IMAGE_SIZE]
                               [--shuffle_train BOOLEAN_FLAG]
                               [--shuffle_test BOOLEAN_FLAG]
                               [--flip BOOLEAN_FLAG]
                               [--num_joints_half_body NUM_JOINTS_HALF_BODY]
                               [--prob_half_body PROB_HALF_BODY]
                               [--rot_factor ROT_FACTOR]
                               [--scale_factor SCALE_FACTOR]
                               [--train_set TRAIN_SET]
                               [--test_set TEST_SET]
                               [--model_name MODEL_NAME]
                               [--detector_name DETECTOR_NAME]
                               [--detector_type DETECTOR_TYPE]
                               [--num_epochs NUM_EPOCHS]
                               [--learning_rate LEARNING_RATE]
                               [--learning_rate_factor LEARNING_RATE_FACTOR]
                               [--patience PATIENCE]
                               [--batch_size BATCH_SIZE]
                               [--save_frequency SAVE_FREQUENCY]
                               [--optimizer OPTIMIZER]
                               [--momentum MOMENTUM]
                               [--nesterov BOOLEAN_FLAG]
                               [--gamma1 GAMMA1]
                               [--gamma2 GAMMA2]
                               [--bbox_thr BBOX_THR]
                               [--det_nms_thr DET_NMS_THR]
                               [--img_thr IMG_THR]
                               [--in_vis_thr IN_VIS_THR]
                               [--nms_thr NMS_THR]
                               [--oks_thr OKS_THR]
                               [--use_gt_bbox USE_GT_BBOX]
```  


Despite being a long list of parameters, only *EXP_DIRECTORY* and *DATASET_NAME* are required. Check the [CONFIG File](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/CONFIG.py) to see the deault parameters.


**Note**: run `python 01_create_experiment.py --help` to display a more detailed description of the parameters.


## Person Detection

### Training a Person Detector

The script `02_train_faster_rcnn.py` can be used to train a person detector model.

```shell
$ python 02_train_faster_rcnn.py --help]
  usage: 02_train_faster_rcnn.py [-h] [-d EXP_DIRECTORY]
                               [--checkpoint CHECKPOINT]
                               [--dataset_name DATASET_NAME]
                               [--perceptual_loss PERCEPTUAL_LOSS]
                               [--drop_head DROP_HEAD] [--save SAVE]
                               [--alpha ALPHA] [--styles STYLES]
                               [--percentage PERCENTAGE]
                               [--resume_training RESUME_TRAINING]
```

For the task of person detection, we use the Faster R-CNN model provided by [Torchvision](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

The `--checkpoint CHECKPOINT` parameter can be used to load the parameters of a pretrained model for fine-tuning. The *.pth* file contianing the pretrained parameters **must** be in the /EXP_DIRECTORY/models/detector/ directory.

Running the training script will create the *detector_logs.json* file, where training loss and validation mAP results will be stored after every epoch. Furthermore, every few epochs (specified by the `--save_frequency` parameter) a checkpoint will be saved in the /EXP_DIRECTORY/models/detector/ directory.

The following two examples show how to train a Faster R-CNN from scratch on MS-COCO and how to tune a Faster RCNN on ClassArch from a pretrained model respectively.

```shell
$ python 02_train_faster_rcnn.py -d ExperimentFasterRCNN --dataset_name coco
$ python 02_train_faster_rcnn.py -d ExperimentFasterRCNN --dataset_name arch_data --checkpoint file_with_pretrained_parameters.pth
```


### Evaluating a Person Detector

The script `03_evaluate_faster_rcnn.py` can be used to evaluate a person detector model.


```shell
$ python 03_evaluate_faster_rcnn.py --help
  usage 03_evaluate_faster_rcnn.py [-h] [-d EXP_DIRECTORY]
                             [--checkpoint CHECKPOINT]
                             [--dataset_name DATASET_NAME]
                             [--perceptual_loss PERCEPTUAL_LOSS]
                             [--drop_head DROP_HEAD] [--save SAVE]
                             [--alpha ALPHA] [--styles STYLES]
                             [--percentage PERCENTAGE]
                             [--resume_training RESUME_TRAINING]
```

This script works very similarly to the training pipeline, but it will just evaluate a model in the test set.

The `--checkpoint CHECKPOINT` parameter points to the *.pth* file containing the parameters of the model to evaluate. This *.pth* file **must** be in the /EXP_DIRECTORY/models/detector/ directory.

The `--dataset_name` parameter defines the dataset that will be used for evaluation. If this parameter is not specified, the model will be evaluated on the dataset defined in the `experiment_parameters.json` file.

If parameter `--save` is set to True: `--save True`, the obtained detections will be stored under the */plots* directory.

The following example shows how to evaluate a trained Faster R-CNN on the COCO and Styled-COCO datasets respectively

```shell
$ python 03_evaluate_faster_rcnn -d ExperimentFasterRCNN --checkpoint checkpoint_epoch_final.pth --dataset_name coco
$ python 03_evaluate_faster_rcnn -d ExperimentFasterRCNN --checkpoint checkpoint_epoch_final.pth --dataset_name styled_coco
```

### Relevant Files

In this section, we list some relevant functions and files for the person detection functionalities:

 - **[lib/model_setup.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/lib/model_setup.py)**: Contains methods for initializing the person detector models and to save/load the checkpoints.

- **[lib/bounding_box.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/lib/bounding_box.py)**: Methods for filtering and post-processing the detections from the models.

- **[data/data_loaders.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/data/data_loaderls.py)**: Contains the methods used for loading the datasets for the task of person detection. Furthermore, also takes care of the training/validation/test splitting.


### Train/Evaluate on your Data

Some changes need to be made in order to adapt the trainint/evaluation pipelines to new datasets.

  1. Change the paths from the [CONFIG File](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/CONFIG.py) to paths in your file system.

  2. Use the [`Detection_Dataset.py`](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/data/Detection_Dataset.py) class to fit your data. The `Detection_Dataset.py` module receives the paths to the image and annotation direrctory, and then loads the data for the task of person detection using the `pycocotools`interface. For this module to work, the annotations **must** follow the COCO format.

  If your do not have COCO-like annotations, modify the [`ArchDataset.py`](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/data/ArchDataset.py) module to adapt your data.

  3. Add your dataset to the `data/data_loaders.py` `get_detection_dataset()` method to access it from the training and evaluation pipelines.


## Pose Estimation

The task of pose estimation consists of detecting certain body joint locations, denoted as keypoints, and combining the to form pose skeletons. In our work, we use the HRNet model to detect the keypoints.

### Training HRNet

The training of the HRNet keypoint detector model can be performed with the `02_train.py` script. Regarding the model implementation, we adapt the code and use the pretrained parameters from [this repository](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).


```shell
$ python 02_train.py --help
usage 02_train_faster_rcnn.py [-h] [-d EXP_DIRECTORY]
                               [--checkpoint CHECKPOINT]
                               [--dataset_name DATASET_NAME]
                               [--perceptual_loss PERCEPTUAL_LOSS]
                               [--drop_head DROP_HEAD] [--save SAVE]
                               [--alpha ALPHA] [--styles STYLES]
                               [--percentage PERCENTAGE]
                               [--resume_training RESUME_TRAINING]
```

By default, the HRNet model is initialized with pretrained weights that achieve SOTA results for the MS-COCO dataset. However, different initial parameters can be loaded using the `--checkpoint CHECKPOINT` argument. This argument points to a *.pth* file contained in the /EXP_DIRECTORY/models/ directory.

Running this training script will create a *training_logs.json* file, in which loss, accuracy and mAP values will be stored for training and validation epochs. Furthermore, every `--save_frequency` epochs, a new checkpoint will be saved under the /EXP_DIRECTORY/models/ directory.

If parameter `--save` is set to True: `--save True`, the obtained pose skeletons will be stored under the */plots* directory.


The following examples would start training an HRNet model from scratch and from a checkpoint saved after 100 epochs respectively.

```shell
$ python 02_train.py -d ExperimentHRNet
$ python 02_train.py -d ExperimentHRNet --checkpoint checkpoint_epoch_100.pth
```

### Evaluating HRNet

The evaluation procedure of the HRNet model is very similar to the training one, and can be performed with the `03_evaluate.py` script.

```shell
$ python 03_evaluate.py --help
  usage 03_evaluate.py [-h] [-d EXP_DIRECTORY]
                         [--checkpoint CHECKPOINT]
                         [--dataset_name DATASET_NAME]
                         [--perceptual_loss PERCEPTUAL_LOSS]
                         [--drop_head DROP_HEAD] [--save SAVE]
                         [--alpha ALPHA] [--styles STYLES]
                         [--percentage PERCENTAGE]
                         [--resume_training RESUME_TRAINING]
```

The `--checkpoint CHECKPOINT` parameter points to the .pth file containing the parameters of the model to evaluate. This .pth file must be in the /EXP_DIRECTORY/models/ directory.

The `--dataset_name` parameter defines the dataset that will be used for evaluation. If this parameter is not specified, the model will be evaluated on the dataset defined in the `experiment_parameters.json` file.

The following example shows how to evaluate a trained HRNet on the COCO and Styled-COCO datasets respectively

```shell
$ python 03_evaluate.py -d ExperimentHRNet --checkpoint checkpoint_epoch_trained.pth --dataset_name coco
$ python 03_evaluate.py -d ExperimentHRNet --checkpoint checkpoint_epoch_trained.pth --dataset_name styled_coco
```

### Relevant Files

In this section, we list some of the relevant lib-files that are used for the keypoint detection and pose estimation functionalities.


 - **[lib/model_setup.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/lib/model_setup.py)**: Contains methods for initializing the HRNet model and to save/load the checkpoints.

- **[data/data_loaders.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/data/data_loaderls.py)**: Contains the methods used for loading the datasets for the task of pose estimation. Furthermore, also takes care of the training/validation/test splitting.

- **[lib/pose_parsing.py](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/lib/pose_parsing.py)**: Methods for extracting joint coordinates from predicted heatmaps, for creating pose vectors from keypoint detections or to convert detections from detected person instance to image coordinates.
