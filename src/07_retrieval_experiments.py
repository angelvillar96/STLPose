"""
Performing a retrieval experiment for the arch_data dataset. It works as follows:
    - For each query, we obtaine the top-N retrievals for a certain metric
    - We measure retrieval performance using several approaches

@author: Angel Villar-Corrales 
"""

import os
import copy
import json
import argparse
from time import time
from tqdm import tqdm

import numpy as np

from lib.arguments import process_retrieval_arguments
from lib.logger import Logger, log_function, print_
from lib.metrics import score_retrievals
from lib.pose_database import load_database, load_knn, process_pose_vector, \
                              get_neighbors_idxs
import lib.pose_parsing as pose_parsing
from lib.utils import create_directory, for_all_methods, load_experiment_parameters, \
                      load_character_narrative_maps, timestamp
from lib.visualizations import draw_skeleton
import CONSTANTS


@for_all_methods(log_function)
class RetrievalExp:
    """
    Class implementing all logic for the retrival experiments

    Args:
    -----
    exp_directory: string
        path to the experiment directory
    params: Namespace
        namespace containing the command line arguments:  'dataset_name', 'approach,
        'normalize', 'num_retrievals', 'num_exps', 'retrieval_method', 'penalization',
        'shuffle'
    """

    def __init__(self, exp_directory, params):
        """
        Initializer of the experiment object
        """

        self.exp_directory = exp_directory
        self.params = params

        self.knn, self.database, self.features = None, None, None
        self.key_list, self.n_entries = [], 0
        self.char_to_narr, self.narr_to_char = {}, {}

        self._load_resources()

        return


    def _load_resources(self):
        """
        Loading preprocessed data and kNN for retrieval purposes
        """

        # loading database and retrieval resources
        knn, database, features = load_knn(database_file=self.params.database_file)
        keys_list = list(database.keys())
        n_entries = len(keys_list)

        self.knn, self.database, self.features = knn, database, features
        self.key_list, self.n_entries = keys_list, n_entries
        if(self.params.num_retrievals < 0):
            self.params.num_retrievals = self.n_entries

        self.char_to_narr, self.narr_to_char = load_character_narrative_maps()

        return


    def retrieval_experiment(self):
        """
        Performing a retrieval experiment by iterating over all poses and evaluating
        the accuracy/precision of the retrieved poses
        """

        preprocess_pose = lambda x: process_pose_vector(vector=x,
                                                        approach=self.params.approach,
                                                        normalize=self.params.normalize)

        character_results = []
        narrative_results = []
        start = time()
        for key_idx, key in enumerate(tqdm(self.key_list)):
            # fetching and preprocessing query data
            query = self.database[key]
            query_img = query["img"]
            query_joints = query["joints"].numpy()
            label_character = query["character_name"]
            label_narrative = self.char_to_narr[label_character]
            pose_vector = preprocess_pose(query_joints)

            # retrieving and raking using a similarity/metric approach
            idx, dists = get_neighbors_idxs(pose_vector, k=self.params.num_retrievals,
                                            num_retrievals=self.params.num_retrievals,
                                            approach=self.params.approach, knn=self.knn,
                                            retrieval_method=self.params.retrieval_method,
                                            penalization=self.params.penalization,
                                            database=self.features)
            retrievals = [self.database[self.key_list[j]] for j in idx]

            # extracting information/labels from retrievals
            retrieved_characters = [r["character_name"] for r in retrievals]
            retrieved_narratives = [self.char_to_narr[c] for c in retrieved_characters]

            character_metrics = score_retrievals(label=label_character,
                                                 retrievals=retrieved_characters)
            narrative_metrics = score_retrievals(label=label_narrative,
                                                 retrievals=retrieved_narratives)
            character_results.append(character_metrics)
            narrative_results.append(narrative_metrics)

            # if(key_idx > self.params.num_exps):
                # break

        end = time()
        self.elapsed_time = end - start
        self.character_results = character_results
        self.narrative_results = narrative_results

        return


    def process_retrieval_results(self, type="character", save=True):
        """
        Post-processing retrieval results obtaining detailed metrics at different
        level: character level, narrative level, character-wise and narrative-wise
        """

        assert type in ["character", "narrative"], "Only ['character', 'narrative'] "\
            "result types are allowed..."
        scores = self.character_results if type == "character" else self.narrative_results
        res_template = {
            "p@1": [],
            "p@5": [],
            "p@10": [],
            "p@rel": [],
            "mAP": [],
            "r@1": [],
            "r@5": [],
            "r@10": [],
            "r@rel": [],
            "mAR": [],
        }
        results = {"general": copy.deepcopy(res_template)}

        # grouping results given labels
        for score in scores:
            cur_label = score["label"]
            # initializng dictionary if necessaty
            if(cur_label not in results):
                results[cur_label] = copy.deepcopy(res_template)
            # appending results to corresponding dictionaries
            for key in res_template.keys():
                results[cur_label][key].append(score[key])
                if(score[key] >= 0):  # avoinding -1s, which correspond one-instance classes
                    results["general"][key].append(score[key])

        # obtaining averaging results
        for result_key in results.keys():
            for key in results[result_key].keys():
                results[result_key][key] = np.mean(results[result_key][key])
            print_(f"Retrieval results for {type}: '{result_key}'")
            print_(results[result_key])
        print_("\n")

        # saving into a dictionary in the experiment directory
        dataset_name = self.params.database_file.split("database_")[1].split("_eval")[0]
        savedict = {
            "results": results,
            "metadata": {
                "timestamp": timestamp(),
                "dataset_name": dataset_name,
                "retrival_time": self.elapsed_time,
                "database size": self.n_entries,
                "retrieval_level": type,
                "retrieval_method": self.params.retrieval_method,
                "pose approach": self.params.approach,
                "missing kpt penalization": self.params.penalization,
                "normalized poses": self.params.normalize
            }
        }
        fname = f"retrieval_results_type_{type}_method_{self.params.retrieval_method}_" \
                f"approach_{self.params.approach}_penalization_{self.params.penalization}_" \
                f"normalized_{self.params.normalize}.json"
        fpath = os.path.join(self.exp_directory, fname)
        with open(fpath, "w") as file:
            json.dump(savedict, file)

        return


if __name__ == "__main__":
    os.system("clear")
    args = process_retrieval_arguments()

    logger = Logger(args.exp_directory)
    message = f"Initializing retrieval experiment..."
    logger.log_info(message=message, message_type="new_exp")
    logger.log_params(params=vars(args))

    # retrieval_experiment
    retriever = RetrievalExp(exp_directory=args.exp_directory, params=args)
    retriever.retrieval_experiment()
    retriever.process_retrieval_results(type="character", save=True)
    retriever.process_retrieval_results(type="narrative", save=True)
    logger.log_info(message="Retrieval experiment completed successfully")


#
