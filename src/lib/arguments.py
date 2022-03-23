"""
Methods for reading and processing command line arguments

EnhancePoseEstimation/src/lib
@author: Angel Villar-Corrales 
"""


import os
import ast
import argparse
from argparse import Namespace
from CONFIG import CONFIG


def process_create_experiment_arguments():
    """
    Processing command line arguments for 01_create_experiment script
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment" +\
                        "folder will be created", required=True, default="test_dir")

    # dataset parameters
    parser.add_argument('--dataset_name', help="Dataset to take the images from " +\
                        "[coco, styled_coco, arch_data, 'combined'", required=True, default="styled_coco")
    parser.add_argument('--alpha', help="Identifier of the Styled-COCO dataset ['random', '1.0'" \
                        "'0.5'].", default="0.5")
    parser.add_argument('--styles', help="Styles transfered for Styled-COCO ['redblack', "\
                        "'scenes'].", default="redblack")
    parser.add_argument('--image_size', help="Size used to standarize the images (size x size)")
    parser.add_argument('--shuffle_train', help="If True, train set is iterated randomly",
                        action='store_true')
    parser.add_argument('--shuffle_test', help="If True, valid/test set is iterated randomly",
                        action='store_true')
    parser.add_argument('--flip', help="If True, images might be flippled during training." +\
                        " We recommend setting it to true", action='store_true')
    parser.add_argument('--num_joints_half_body', help="Number of joints in the 'half-body'")
    parser.add_argument('--prob_half_body', help="Probabilitiy of considering only half the body")
    parser.add_argument('--rot_factor', help="Maximum rotation angle for the affine " +\
                        " transofrm. A suiable value is 45", type=float)
    parser.add_argument('--scale_factor', help="Maximum scaling factor for the affine " +\
                        "transofrm. A suitalbe value is 0.35", type=float)
    parser.add_argument('--train_set', help="Name of the train set to use, i.e., 'train2017'")
    parser.add_argument('--test_set', help="Name of the test/valid set to use, i.e., 'val2017'")

    # model parameters
    parser.add_argument('--model_name', help="Model to use for Human Pose Estimation " +\
                        "[OpenPose, OpenPoseVGG, HRNet]", default="HRNet")
    parser.add_argument('--detector_name', help="Name of the person detector model to use "\
                         "prior to the pose estimation. ['faster_rcnn', 'efficientdet']",
                         default="faster_rcnn")
    parser.add_argument('--detector_type', help="Type of EfficientDet model used for the" +\
                        " person detection ['', 'd0', 'd3']", default='')

    # training parameters
    parser.add_argument('--num_epochs', help="Number of epochs to train for", type=int)
    parser.add_argument('--learning_rate', help="Learning rate", type=float)
    parser.add_argument('--learning_rate_factor', help="Factor to drop the learning rate " +\
                        "when metric does not further improve", type=float)
    parser.add_argument('--scheduler', help="Learning rate scheduler", default="plateau")
    parser.add_argument('--patience', help="Patience factor of the lr scheduler", type=int)
    parser.add_argument('--batch_size', help="Number of examples in each batch", type=int)
    parser.add_argument('--save_frequency', help="Number of epochs after which we save " +\
                        "a checkpoint", type=int)
    parser.add_argument('--optimizer', help="Method used to update model parameters")
    parser.add_argument('--momentum', help="Weight factor for the momentum", type=float)
    parser.add_argument('--nesterov', help="If True, Nesterovs momentum is applied", action='store_true')
    parser.add_argument('--gamma1', help="Gamma1 variable of the adam optimizer", type=float)
    parser.add_argument('--gamma2', help="Gamma2 variable of the adam optimizer", type=float)
    parser.add_argument('--perceptual_loss',  help='If True, training of styled-coco '+\
                        'takes into account the perceptual loss', action='store_true')
    parser.add_argument('--perceptual_weight',  help='Approach used for weighting the '+\
                        'perceptual_loss: ["add", "lambda"]', default="add")
    parser.add_argument('--lambda_D', help="Lambda value weighting model loss", type=float)
    parser.add_argument('--lambda_P', help="Lambda value weighting perceptual loss", type=float)

    # parsing and evaluation parameters
    parser.add_argument('--bbox_thr', help="Threshold for the bbox confidence in " +\
                        "Faster RCNN", type=float)
    parser.add_argument('--det_nms_thr', help="Threshold that decideds if two bboxes correspond "\
                        "to the same detection for NMS", type=float)
    parser.add_argument('--img_thr', help="TODO", type=float)
    parser.add_argument('--in_vis_thr', help="TODO", type=float)
    parser.add_argument('--nms_thr', help="Threshold that decideds if two detections correspond "\
                        "to the same keypoint for NMS", type=float)
    parser.add_argument('--oks_thr', help="Threshold to consider a keypoint in the OKS", type=float)
    parser.add_argument('--use_gt_bbox', help="If True, precomputed detections are " +\
                        "used for evaluation", action='store_true')

    args = parser.parse_args()

    # enforcing correct values
    assert args.dataset_name in ["coco", "styled_coco", "arch_data", "combined"],\
        "Wrong dataset given. Only ['coco', 'styled_coco', 'arch_data', 'combined'] are allowed"
    assert args.model_name in ["OpenPose", "OpenPoseVGG", "HRNet"],\
        "Wrong model name given. Only ['OpenPose', 'OpenPoseVGG', 'HRNet'] are allowed"
    assert args.detector_name in ['faster_rcnn', 'efficientdet'], \
        "Wrong detector name given. Only ['faster_rcnn', 'efficientdet'] are allowed"
    assert args.detector_type in ['', 'd0', 'd3'], \
        "Wrong detector type given. Only [None, 'd0', 'd3'] are allowed"
    assert args.alpha in ["random", "0.5", "1.0"], \
        "Wrong alpha parameter. Only ['random', '1.0', '0.5'] are accepted"
    assert args.styles in ["redblack", "scenes"], \
        "Wrong styles parameter. Only ['redblack', 'scenes'] are accepted"
    assert args.perceptual_weight in ["add", "lambda"], \
        "Wrong 'perceptual_weight' value. Only ['add', 'lambda'] are allowed"

    return args


def get_directory_argument(get_checkpoint=False, get_dataset=False, get_perceptual_flag=False):
    """
    Reading the directory passed as an argument
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()
    # important four parameters
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory")
    parser.add_argument("--checkpoint", help="Name of the checkpoint file to load")
    parser.add_argument("--dataset_name", help="Name of the dataset to use for training" \
                        " or evaluation purposes ['coco', 'styled_coco', 'arch_data']",
                        default="")
    parser.add_argument('--perceptual_loss', help='If training styled-coco with perceptual loss',
                        type=bool, default=False)

    # parameters relevant for tuning/evaluation
    parser.add_argument("--drop_head", help="If True, classification/regression head of the "\
                        "faster R-CNN model will be re-initialized for transfer-learning")
    parser.add_argument("--save", help="If True, images from the evaluations will be saved")
    parser.add_argument('--alpha', help="Identifier of the Styled-COCO dataset ['random', '1.0'" \
                        "'0.5'].")
    parser.add_argument('--styles', help="Styles transfered for Styled-COCO ['redblack', "\
                        "'scenes'].")
    parser.add_argument('--percentage', help="When training on ArchData, percentage of the total "\
                        "image used for training (e.g. 50 or 75)", type=float)
    parser.add_argument("--resume_training", help="If True, training is continued where " +\
                        "checkpoint left it. Otherwise, training is started from scratch "+\
                        "but with the weights of the checkpoit")

    args = parser.parse_args()

    exp_directory = args.exp_directory
    checkpoint = args.checkpoint
    dataset_name = args.dataset_name

    assert args.alpha in [None, "random", "0.5", "1.0"], \
        "Wrong alpha parameter. Only ['random', '1.0', '0.5'] are accepted"
    assert args.styles in [None, "redblack", "scenes"], \
        "Wrong styles parameter. Only ['redblack', 'scenes'] are accepted"
    assert args.percentage is None or (args.percentage >=1 and args.percentage <= 100), \
        "ERROR! 'Percentage' parameter must be in range [1, 100]"

    params = {
        "save": (args.save == "True") if args.save is not None else False,
        "resume_training": (args.resume_training == "True")
            if args.resume_training is not None else False,
        "drop_head": (args.drop_head == "True") if args.drop_head is not None else False,
        "use_perceptual_loss": args.perceptual_loss,
        "alpha": args.alpha,
        "styles": args.styles,
        "percentage": args.percentage
    }
    params = Namespace(**params)

    # making sure experiment directory and checkpoint file exist
    exp_directory = process_experiment_directory_argument(exp_directory)
    if(get_checkpoint is True and checkpoint is not None):
        checkpoint = process_checkpoint(checkpoint, exp_directory)
    if(get_dataset is True):
        assert dataset_name in ["", "coco", "styled_coco", "arch_data", "combined"]
        dataset_name = None if dataset_name == "" else dataset_name

    if(get_dataset==True and get_checkpoint==True):
        return exp_directory, checkpoint, dataset_name, params
    elif(get_dataset):
        return exp_directory, dataset_name, params
    elif(get_checkpoint):
        return exp_directory, checkpoint, params
    return exp_directory, params



def process_retrieval_arguments():
    """
    Processing command line arguments
    """

    # defining command line arguments
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument("-d", "--exp_directory", help="Path to the root of the " +\
                        "experiment folder", required=True, default="test_dir")

    # dataset parameters
    parser.add_argument("--database_file", help="Name of the file contaning the "\
                        "preprocessed database", required=True)
    parser.add_argument("--approach", help="Approach used to consider similarity.\n"\
                        "    'upper_body': only considering upper-body keypoints.\n"\
                        "    'full_body': considering lower and upper body (not head).\n"\
                        "    'all_kpts': considering all keypoints.", default="full_body")
    parser.add_argument("--normalize", help="If True, pose vectors are L-2 normalized.",
                        default="True")

    # retrtieval parameters
    parser.add_argument("--num_retrievals", help="Number of elements to retrieve from "\
                        "the database. -1 means all", type=int, default=-1)
    parser.add_argument("--num_exps", help="Number of times to repeat the experiment.",
                        type=int, default=5)
    parser.add_argument("--retrieval_method", help="Method used to compute the simmilarity "\
                        "between query and database elements. ['knn', 'euclidean_distance', " \
                        "'manhattan_distance','cosine_similarity', 'confidence_score'," \
                        "'oks_score']", default="knn")
    parser.add_argument("--penalization", help="Startegy for penalizing points that are not "\
                        "present in the image. ['none', 'zero_coord', 'mean', 'max']",
                        default="zero_coord")
    parser.add_argument("--shuffle", help="Boolean. If activated, query skeletons are"\
                        "sampled at random", default="False")


    args = parser.parse_args()

    # procesing command line arguments to enforce correct values
    args.exp_directory = process_experiment_directory_argument(args.exp_directory)#
    assert args.database_file[:4] == "data", "ERROR! DB_File must start with 'data'"
    assert os.path.exists(os.path.join(CONFIG['paths']["knn_path"], args.database_file))
    args.normalize = (args.normalize=="True")
    args.shuffle = (args.shuffle=="True")
    assert args.retrieval_method in ['knn', 'euclidean_distance', 'manhattan_distance',
        'cosine_similarity', 'confidence_score', 'oks_score']
    assert args.penalization in ["none", "zero_coord", "mean", "max"]
    assert args.approach in ['upper_body', 'full_body', 'all_kpts']

    return args


def process_experiment_directory_argument(exp_directory):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """

    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory)):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()

    return exp_directory


def process_checkpoint(checkpoint, exp_directory):
    """
    Making sure the checkpoint to load exists
    """

    # checkpoint = None corresponds to untrained model
    if(checkpoint is None):
        return checkpoint

    checkpoint_path = os.path.join(exp_directory, "models", checkpoint)
    checkpoint_path_det = os.path.join(exp_directory, "models", "detector", checkpoint)
    if(not os.path.exists(checkpoint_path) and not  os.path.exists(checkpoint_path_det)):
        print(f"ERROR! Checkpoint {checkpoint_path} does not exist...")
        print(f"ERROR! Checkpoint {checkpoint_path_det} does not exist either...")
        print(f"     The given checkpoint was: {checkpoint}")
        print("\n\n")
        exit()

    return checkpoint

#
