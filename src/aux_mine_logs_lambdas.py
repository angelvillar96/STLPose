"""
Mining the experiment logs to extract the Optimized Lambda values from the
Optuna parameter search
"""

import os
import json
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

from lib.arguments import get_directory_argument


def is_instersting(line):
    """
    """
    interesting_lines = ["lambda_D", "lambda_P", "Valid Accuracy"]
    for l in interesting_lines:
        if(l in line):
            return l
    return None

def main():
    """
    Main logic
    """

    # processing command line arguments
    exp_path, _ = get_directory_argument()
    logs_path = os.path.join(exp_path, "logs.txt")
    plots_path = os.path.join(exp_path, "plots")
    savepath = os.path.join(exp_path, "lambdas_logs.json")
    if(not os.path.exists(logs_path)):
        print(f"ERROR! 'logs.txt' file in path {exp_path} does not exist...")
        exit()

    # getting all lines from the logs-gile
    with open(logs_path) as f:
        content = f.readlines()

    # mining relevant information from logs
    interesting_lines = ["lambda_D", "lambda_P", "Valid Accuracy"]
    exps = {}
    counter = 0
    cur_exp = 0
    for c in tqdm(content):
        val = is_instersting(c)
        if(val is None):
            continue
        elif(val == "lambda_D"):
            cur_exp += 1
            exps[cur_exp] = {}
            exps[cur_exp]["lambda_D"] = float(c.split("'")[-2])
        elif(val == "lambda_P"):
            exps[cur_exp]["lambda_P"] = float(c.split("'")[-2])
        elif(val == "Valid Accuracy"):
            if(counter == 4):
                exps[cur_exp]["accuracy"] = float(c.split(" ")[-1][:-2])
                counter = 0
            else:
                counter += 1

    # dict to list
    lambdas = []
    accs = []
    for key, data in exps.items():
        if("accuracy" in data):
            accs.append(data["accuracy"])
            lambdas.append(data["lambda_P"])
    best_idx = np.argmax(accs)
    best_acc = accs[best_idx]
    best_lambda = lambdas[best_idx]
    print(f"Best results were achieved for experiments {best_idx + 1}:")
    print(f"    Lambda_P = {best_lambda}")
    print(f"    Accuracy = {best_acc}")

    # plotting
    fig, ax = plt.subplots(1,1)
    sorted_idx = np.argsort(lambdas)
    plt.plot(np.array(lambdas)[sorted_idx], np.array(accs)[sorted_idx], linewidth=3)
    plt.scatter(np.array(lambdas)[sorted_idx], np.array(accs)[sorted_idx], linewidth=3)
    plt.grid()
    plt.xlabel("Lambda_P value")
    plt.title("Validation Acc @ Epoch 5")
    plt.savefig(os.path.join(plots_path, "lambda_exp.png"))

    # saving lambda information into a json file
    with open(savepath, "w") as f:
        json.dump(exps, f)

    return


if __name__ == "__main__":
    os.system("clear")
    main()
