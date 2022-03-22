"""
Loading a precomputed Pickle file containing an Optuna study and visualizing
its contents
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd

import lib.arguments as arguments
import lib.utils as utils
from CONFIG import CONFIG


def process_arguments():
    """
    Processing command line arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment" +\
                        "folder will be created", required=True)
    parser.add_argument("--study", help="Name of the .pkl file to load", required=True)
    args = parser.parse_args()

    exp_directory = args.exp_directory
    exp_directory = arguments.process_experiment_directory_argument(exp_directory)
    study_file = os.path.join(exp_directory, args.study)
    assert os.path.exists(study_file), f"ERROR! Study {study_file} does not exist"

    return exp_directory, study_file


def main():
    """
    Main logic for investigating optuna results
    """

    # processing command line arguments and loading study
    exp_directory, study_file = process_arguments()
    with open(study_file, "rb") as file:
        study = pickle.load(file)

    # reading parameters
    best_trial = study.best_trial
    trials = study.trials

    # dispalying info
    for trial in trials:
        print(f"Trial {trial.number}:")
        print(f"    Lambda_D: {trial.params['Lambda_D']}")
        print(f"    Lambda_P: {trial.params['Lambda_P']}")
        print(f"    Metric: {trial.value}")
    print(f"\nBest Trial: Trial {best_trial.number}")
    print(f"    Best Lambda_D: {best_trial.params['Lambda_D']}")
    print(f"    Best Lambda_P: {best_trial.params['Lambda_P']}")
    print(f"    Best Metric: {best_trial.value}")

    return



if __name__ == "__main__":
    os.system("clear")
    main()
