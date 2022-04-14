"""
Auxiliary methods to handle logs, argument files and other
functions that do not belong to any particular class

@author: Angel Villar-Corrales
"""

import os
import json
import datetime

import numpy as np
from matplotlib import pyplot as plt

from lib.logger import log_function, print_
from CONFIG import CONFIG, DEFAULT_ARGS


@log_function
def create_configuration_file(exp_path, config, args):
    """
    Creating a configuration file for an experiment, including the hyperparemters
    used and other relevant variables

    Args:
    -----
    exp_path: string
        path to the experiment folder
    config: dictionary
        dictionary with the global configuration parameters: paths, ...
    args: Namespace
        dictionary-like object with the data corresponding to the command line arguments

    Returns:
    --------
    exp_data: dictionary
        dictionary containing all the parameters is the created experiments
    """

    exp_data = {}
    exp_data["exp_created"] = timestamp()
    exp_data["last_modified"] = timestamp()
    exp_data["random_seed"] = config["random_seed"]
    exp_data["num_workers"] = config["num_workers"]

    # loading default args
    args_dict = vars(args)

    # dataset parameters
    exp_data["dataset"] = DEFAULT_ARGS["dataset"]
    for key in DEFAULT_ARGS["dataset"]:
        if(args_dict[key] is not None):
            exp_data["dataset"][key] = args_dict[key]

    # model parameters
    exp_data["model"] = DEFAULT_ARGS["model"]
    for key in DEFAULT_ARGS["model"]:
        if(args_dict[key] is not None):
            exp_data["model"][key] = args_dict[key]

    # training parameters
    exp_data["training"] = DEFAULT_ARGS["training"]
    for key in DEFAULT_ARGS["training"]:
        if(args_dict[key] is not None):
            exp_data["training"][key] = args_dict[key]

    # evaluation parameters
    exp_data["evaluation"] = DEFAULT_ARGS["evaluation"]
    for key in DEFAULT_ARGS["evaluation"]:
        if(args_dict[key] is not None):
            exp_data["evaluation"][key] = args_dict[key]

    # creating file and saving it in the experiment folder
    exp_data_file = os.path.join(exp_path, "experiment_parameters.json")
    with open(exp_data_file, "w") as file:
        json.dump(exp_data, file)

    return exp_data


@log_function
def load_experiment_parameters(exp_path):
    """
    Loading the experiment parameters given the path to the experiment directory

    Args:
    -----
    exp_path: string
        path to the experiment directory

    Returns:
    --------
    exp_data: dictionary
        dictionary containing the current experiment parameters
    """

    exp_data_path = os.path.join(exp_path, "experiment_parameters.json")
    with open(exp_data_path) as file:
        exp_data = json.load(file)

    return exp_data


@log_function
def create_directory(path, name=None):
    """
    Checking if a directory already exists and creating it if necessary

    Args:
    -----
    path: string
        path/name where the directory will be created
    name: string
        name fo the directory to be created
    """

    if(name is not None):
        path = os.path.join(path, name)

    if(not os.path.exists(path)):
        os.makedirs(path)

    return


@log_function
def create_train_logs(exp_path):
    """
    Initializing training logs to save loss during trainig and validation

    Args:
    -----
    exp_path: string
        path to the root directory of the experiment

    Returns:
    -------
    training_logs: dictionary
        data structure where the training logs are updated
    """

    training_logs = {}
    training_logs["last_modified"] = timestamp()
    training_logs["iterations"] = 0
    training_logs["loss"] = {}
    training_logs["loss"]["training"] = []
    training_logs["loss"]["validation"] = []
    training_logs["accuracy"] = {}
    training_logs["accuracy"]["training"] = []
    training_logs["accuracy"]["validation"] = []

    logs_file = os.path.join(exp_path, "training_logs.json")
    with open(logs_file, "w") as file:
        json.dump(training_logs, file)

    return training_logs


@log_function
def load_train_logs(exp_path):
    """
    Loading train logs to keep training
    """

    # saving json file
    logs_file = os.path.join(exp_path, "training_logs.json")
    with open(logs_file) as file:
        logs = json.load(file)
    return logs


@log_function
def update_train_logs(exp_path, training_logs, iterations, train_loss, valid_loss,
                      train_acc, valid_acc):
    """
    Updating the training logs and saving the updated version, along with a plot
    with the current training landscape

    Args:
    -----
    exp_path: string
        path to the root directory of the experiment
    training_logs: dictionary
        data structure with the logs up to date
    iterations: integer
        number of training iterations run so far
    train_loss, valid_loss: float
        loss values for the training and validation epochs respectively
    train_loss, valid_loss: float
        accuracy values for the training and validation epochs respectively
    """

    logs_file = os.path.join(exp_path, "training_logs.json")
    plots_loss_path = os.path.join(exp_path, "plots", "loss_landscape.png")
    plots_loss_path_log = os.path.join(exp_path, "plots", "loss_landscape_log.png")
    plots_acc_path = os.path.join(exp_path, "plots", "accuracy_landscape.png")
    plots_acc_path_log = os.path.join(exp_path, "plots", "accuracy_landscape_log.png")

    # updating logs
    training_logs["last_modified"] = timestamp()
    training_logs["iterations"] = iterations
    training_logs["loss"]["training"].append(train_loss)
    training_logs["loss"]["validation"].append(valid_loss)
    training_logs["accuracy"]["training"].append(train_acc)
    training_logs["accuracy"]["validation"].append(valid_acc)

    # saving json file
    with open(logs_file, "w") as file:
        json.dump(training_logs, file)

    # saving loss and accuracy landscape
    train_acc = training_logs["accuracy"]["training"]
    valid_acc = training_logs["accuracy"]["validation"]
    train_loss = training_logs["loss"]["training"]
    valid_loss = training_logs["loss"]["validation"]
    epochs = np.arange(len(train_loss))

    # loss landscape plots
    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, train_loss, label="Train")
    ax.plot(epochs, valid_loss, label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    plt.savefig(plots_loss_path)
    ax.set_yscale("log")
    plt.savefig(plots_loss_path_log)

    # accuracy landscape plots
    fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, train_acc, label="Train")
    ax.plot(epochs, valid_acc, label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best")
    plt.savefig(plots_acc_path)
    ax.set_yscale("log")
    plt.savefig(plots_acc_path_log)

    return


@log_function
def create_detector_logs(exp_path):
    """
    Initializing training logs to save loss during trainig and validation of the
    person detector

    Args:
    -----
    exp_path: string
        path to the root directory of the experiment

    Returns:
    -------
    training_logs: dictionary
        data structure where the training logs are updated
    """

    training_logs = {}
    training_logs["last_modified"] = timestamp()
    training_logs["train_loss"] = []
    training_logs["valid_ap"] = []

    logs_file = os.path.join(exp_path, "detector_logs.json")
    with open(logs_file, "w") as file:
        json.dump(training_logs, file)

    return training_logs


@log_function
def load_detector_logs(exp_path):
    """
    Loading detector logs
    """
    logs_file = os.path.join(exp_path, "detector_logs.json")
    with open(logs_file) as file:
        logs = json.load(file)

    return logs


@log_function
def update_detector_logs(exp_path, training_logs, train_loss, valid_ap):
    """
    Updating the detector logs and saving the updated version
    """

    logs_file = os.path.join(exp_path, "detector_logs.json")

    # updating logs
    training_logs["last_modified"] = timestamp()
    training_logs["train_loss"].append(train_loss)
    training_logs["valid_ap"].append(valid_ap)

    # saving json file
    with open(logs_file, "w") as file:
        json.dump(training_logs, file)

    return


@log_function
def save_evaluation_stats(exp_path, stats, detector=False, dataset_name=None,
                          checkpoint=None, alpha=None, styles=None):
    """
    Saving the evaluation stats (precision and recall) to a .json file

    Args:
    -----
    exp_path: string
        path to the root directory of the experiment
    stats: list
        list with the evaluation results as provided by the cocoapi
    detector: boolean
        if True, stats correspond to a person detection model
    dataset_name: string/None
        If not None, writes the name of the dataset in the filename
    """

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                   'AR .75', 'AR (M)', 'AR (L)']
    dataset_f = "" if(dataset_name is None) else f"_{dataset_name}"

    if(detector):
        stats_file = os.path.join(
                exp_path,
                f"detector_evaluation_stats{dataset_f}_styles_{styles}_alpha_{alpha}.json"
            )
    else:
        stats_file = os.path.join(
                exp_path,
                f"evaluation_stats{dataset_f}_styles_{styles}_alpha_{alpha}.json"
            )
    # pairing stats name with value {name: value}
    if(os.path.exists(stats_file)):
        with open(stats_file, "r") as f:
            evaluation_stats = json.load(f)
    else:
        evaluation_stats = {}
    evaluation_stats[checkpoint] = {}
    for idx, stat in enumerate(stats_names):
        evaluation_stats[checkpoint][stat] = stats[idx]

    with open(stats_file, "w") as f:
        json.dump(evaluation_stats, f)

    return


def reset_predictions_file(exp_path):
    """
    Reseting the predictions file. It is called at the beginning of the validation epoch
    """
    results_path = os.path.join(exp_path, CONFIG["paths"]["submission"])
    if(os.path.exists(results_path)):
        os.remove(results_path)
    return


def update_predictions_file(cur_predictions, exp_path):
    """
    Loading precomputed predicitons, appending new predictions and saving in results file
    """
    results_path = os.path.join(exp_path, CONFIG["paths"]["submission"])
    all_results = load_predictions(results_path)
    all_results = all_results + cur_predictions
    save_predictions(all_results, results_path)
    return


def load_predictions(results_path):
    """
    Loading precomptured predictions from the results file
    """
    if(os.path.exists(results_path)):
        with open(results_path) as file:
            all_results = json.load(file)
    else:
        all_results = []
    return all_results


def save_predictions(pred, results_path):
    """
    Saving predcitions into the results file
    """
    with open(results_path, "w") as file:
        json.dump(pred, file)
    return


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way

    Returns:
    --------
    timestamp: string
        current timestamp in format hh-mm-ss
    """
    timestamp = str(datetime.datetime.now()).split('.')[0] \
                                            .replace(' ', '_') \
                                            .replace(':', '-')
    return timestamp


def for_all_methods(decorator):
    """
    Decorator that applies a decorator to all methods inside a class
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


@log_function
def load_character_narrative_maps():
    """
    Loading the dictionaries mapping character names to their narrative and
    containing all characters that belong to a certain narrative
    """

    dict_path = CONFIG["paths"]["dict_path"]
    char_to_narr_name = "char_narrative_map.json"
    narr_to_char_name = "narrative_char_map.json"
    char_path = os.path.join(dict_path, char_to_narr_name)
    narr_path = os.path.join(dict_path, narr_to_char_name)

    if(not os.path.exists(char_path) or not os.path.exists(narr_path)):
        print_("ERROR! Mapping dictionaries between narratives and characters "\
               "do not exists...")
        print_("Run 'aux_map_characters_to_narratives.py' to generate them...")
        exit()

    with open(char_path) as file:
        char_to_narr = json.load(file)
    with open(narr_path) as file:
        narr_to_char = json.load(file)

    return char_to_narr, narr_to_char


#
