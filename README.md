# Enhancing Human Pose Estimation in Ancient Vase Paintings via Perceptually-grounded Style Transfer Learning

## Work in Progress!

This respository contains the main codebase for the submission: *Enhancing Human Pose Estimation in Ancient Vase Paintings via
Perceptually-grounded Style Transfer Learning* with Paper ID: 8558.


## Contents

 * [1. Getting Started](#getting-started)
 * [2. Directory Structure](#directory-structure)
 * [3. Quick Guide](#quick-guide)
 <!-- * [4. Reproduce Results](#reproduce-results) -->


## Getting Started

### Prerequisites

To get the repository running, you will need several python packages, e.g., PyTorch, OpenCV, or matplotlib.

You can install them all easily and avoiding dependency issues by installing the conda environment file included in the repository. To do so, run the following command from the Conda Command Window:

```shell
$ conda env create -f environment.yml
$ activate enhance_pose_estimation
```

*__Note__:* This step might take a few minutes


## Directory Structure

The following tree diagram displays the detailed directory structure of the project. Directory names and paths can be modified in the [CONFIG File](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/CONFIG.py).

Pretrained models can be downloaded [here](https://www.dropbox.com/sh/q25su31iw456lzp/AAAgXJneYes57uahMloHP2Qwa?dl=0).

```
EnhancePoseEstimation
├── databases/
│   ├── database_coco.pkl
│   └── ...
|
├── experiments/
│   ├── detector_tests/
│   ├── hrnet_tests/
|   └── ...
|
├── knn/
|   ├──data_datasets_['arch_data']_approach_all_kpts_metric_euclidean_distance_norm_True.pkl
│   └── ...
|
├── resources/
|   ├── EfficientDet/
|   ├── HRNet/      
|   └── ...
|
├── src/
│   |── data/
|   |    ├── ArchDataset.py
|   |    ├── data_loaders.py
    |    └── ...
│   |── lib/
|   |    ├── arguments.py
|   |    ├── bounding_box.py
|   |    ├── model_setup.py
|   |    └── ...
│   ├── 01_create_experiment.py
│   ├── 02_train_faster_rcnn.py
│   ├── 02_train.py
│   ├── 03_evaluate.py
|   └── ...
|
├── environment.yml
├── README.md
```

Now, we give a short overview of the different directories:

- **databases/**: This directory contains processed retrieval databases, exported by *05_** scripts into a .pkl file.

- **experiments/**: Directory containing the experiment folders. New experiments created are placed automatically under this directory.

- **kNN/**: This directory contains the different arrays and dictionaries (*data_**, *features_** and *graph_**) required to perform pose based image retrieval.

  - Files starting with **data_*** contain a dictionary including processed pose vectors and metadata for the given database.

  - Files starting with **features_*** contain a numpy array with the processed pose vectors from the give database.

  - **graph_*** files contain a HNSW kNN graph for efficient nearest-neighbor pose retrieval.

- **src/**: Code for the experiments. For a more detailed description of the code structure,  [click here](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/README.md)

  - **data/**: Methods for data loading and preprocessing.

  - **models/**: Classes corresponding to the different neural networks used for person detection and pose estimation.

  - **lib/**: Library methods for different purposes, such as command-line arguments handling, implementation of denoisers as neural network layers or evaluation metrics.


## Quick Guide

Follow this section for a quick guide on how to get started with this repository.
[Click here](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/README.md) for a more comprehensive *Getting Started*

### Creating an Experiment

#### Usage

```shell
$ python 01_create_experiment.py [-h --help]
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

Creating an experiment automatically generates a directory in the specified experiments/EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and subdirectories for the models and plots.

Despite being a long list of parameters, only *EXP_DIRECTORY* and *DATASET_NAME* are required. Check the [CONFIG File](https://github.com/angelvillar96/EnhancePoseEstimation/blob/master/src/CONFIG.py) to see the deault parameters.


**Note**: run `python 01_create_experiment.py --help` to display a more detailed description of the parameters.

#### Example

The following example creates in the directory */experiments/example_experiment* an experiment using the Styled-COCO dataset and a Faster R-CNN for person detection.
Then, a small subset of the data is saved into the /plots directory, along with some annotations for person detection and pose estimation.

```shell
$ python 01_create_experiment.py -d example_experiment --dataset_name styled_coco \
    --detector_name faster_rcnn
$ python aux_generate_subset_data.py -d example_experiment/experiment_2020-10-05_09-44-54
```


### Training and Evaluation

Once the experiment is initialized, the models can be trained and evaluated for person detection or pose estimation respectively using the following commands.

```shell
# training and evaluating person detector (Faster R-CNN or EfficientDet)
$ CUDA_VISIBLE_DEVICES=0 python 02_train_faster_rcnn.py -d YOUR_EXP_DIRECTORY
$ CUDA_VISIBLE_DEVICES=0 python 03_evaluate_faster_rcnn.py -d YOUR_EXP_DIRECTORY (--save True)
# training and evaluating keypoint detector (HRNet)
$ CUDA_VISIBLE_DEVICES=0 python 02_train.py -d YOUR_EXP_DIRECTORY
$ CUDA_VISIBLE_DEVICES=0 python 03_evaluate.py -d YOUR_EXP_DIRECTORY (--save True)
```

First, the model will be trained and validated for the number of epochs specified in the configuration file (100 by default). Every 10 epochs, a model checkpoint will be saved under the */models* directory, and every epoch the current loss and metrics will be stored in a *training_logs.json* file.

Then, each of the checkpoints will be evaluated on the test set. These evaluation results are also saved in the *training_logs.json* file. If parameter SAVE is set to True: `--save True`, the detections or pose predictions will be stored under the */plots* directory.



## Citing

Please consider citing if you find our findings or our repository helpful.
```
@article{madhu2020enhancing,
  title={Enhancing human pose estimation in ancient vase paintings via perceptually-grounded style transfer learning},
  author={Madhu, Prathmesh and Villar-Corrales, Angel and Kosti, Ronak and Bendschus, Torsten and Reinhardt, Corinna and Bell, Peter and Maier, Andreas and Christlein, Vincent},
  journal={arXiv preprint arXiv:2012.05616},
  year={2020}
}
```

## Contact

This work has been developed by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php) and  supervised by
 [Prathmesh Madhu](https://lme.tf.fau.de/person/madhu/).

In case of any questions or problems regarding the project or repository, do not hesitate to contact me at villar@ais.uni-bonn.de.
