"""
Initializing a new experiment

@author: Angel Villar-Corrales
"""

import os

import lib.arguments as arguments
import lib.utils as utils
from CONFIG import CONFIG


def create_experiment():
    """
    Logic for creating a new experiment: reading command-line arguments, creating directories
    and generating configuration and parameter files
    """

    # processing command-line arguments
    args = arguments.process_create_experiment_arguments()

    # creating relevant directories for the experiment
    exp_name = f"experiment_{utils.timestamp()}"
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], args.exp_directory, exp_name)
    utils.create_directory(exp_path)
    utils.create_directory(exp_path, "models")
    utils.create_directory(exp_path, "plots")
    utils.create_directory(CONFIG["paths"]["experiments_path"], "offline-resources")

    # creating experiment config file
    utils.create_configuration_file(exp_path=exp_path, config=CONFIG, args=args)

    return

if __name__ == "__main__":

    os.system("clear")
    create_experiment()

#
