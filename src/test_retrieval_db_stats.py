"""
Displaying the statistics of a retrieval dataset

@author: Angel Villar-Corrales 
"""

import os, shutil
import argparse
import pickle
from tqdm import tqdm

import cv2
import numpy as np

from lib.arguments import process_retrieval_arguments
from lib.pose_database import load_database, load_knn
from lib.utils import load_character_narrative_maps

import CONSTANTS
from CONFIG import CONFIG

def main():
    """
    Main logic
    """

    # processing command line arguments
    args = process_retrieval_arguments()
    database_file = args.database_file

    # loading database and retrieval resources
    knn, database, features = load_knn(database_file=database_file)
    keys_list = list(database.keys())
    char_to_narr, _ = load_character_narrative_maps()

    # computing stats
    characters = {}
    narratives = {}
    for i, key in enumerate(keys_list):
        query = database[key]
        cur_character = query["character_name"]
        cur_narrative = char_to_narr[cur_character]
        if(cur_character not in characters):
            characters[cur_character] = 0
        characters[cur_character] += 1
        if(cur_narrative not in narratives):
            narratives[cur_narrative] = 0
        narratives[cur_narrative] += 1

    # Displaying stats
    print("Retrieval Database Stats")
    print("------------------------")
    print(f"N-Samples: {len(keys_list)}")
    print(f"Character Statistics:")
    for char, n in characters.items():
        print(f"    {char}: {n}")
    print(f"Narrative Statistics:")
    for nar, n in narratives.items():
        print(f"    {nar}: {n}")

    return


if __name__ == "__main__":
    os.system("clear")
    main()
