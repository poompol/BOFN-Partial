from typing import Callable, List, Optional

import os
import sys
import torch
from bofn_pt.bofn_pt_trial import bofn_pt_trial
def experiment_manager(
        problem: str,
        first_batch_algo: str,
        second_batch_algo: str,
        Constant_Liar: str,
        first_trial: int,
        last_trial: int,
        n_init_evals: int,
        n_bo_iter: int,
        function_network: Callable,
        input_dim: int,
        network_to_objective_transform: Callable,
        n_first_batch: int or None,
        n_second_batch: int
) -> None:
        # Get Directory and creat directory for results

        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

        if first_batch_algo is not None:
                results_folder = script_dir + "/results/" + problem + "/" + first_batch_algo +"-"+ Constant_Liar + "_" + second_batch_algo + "/"
        else:
                results_folder = script_dir + "/results/" + problem + "/" + second_batch_algo + "/"

        if not os.path.exists(results_folder):
                os.makedirs(results_folder)
        if not os.path.exists(results_folder + "runtimes/"):
                os.makedirs(results_folder + "runtimes/")
        if not os.path.exists(results_folder + "X/"):
                os.makedirs(results_folder + "X/")
        if not os.path.exists(results_folder + "network_output_at_X/"):
                os.makedirs(results_folder + "network_output_at_X/")
        if not os.path.exists(results_folder + "objective_at_X/"):
                os.makedirs(results_folder + "objective_at_X/")

        for trial in range(first_trial,last_trial+1):
                bofn_pt_trial(
                        problem = problem,
                        function_network = function_network,
                        network_to_objective_transform = network_to_objective_transform,
                        input_dim = input_dim,
                        first_batch_algo = first_batch_algo,
                        second_batch_algo = second_batch_algo,
                        Constant_Liar = Constant_Liar,
                        n_init_evals = n_init_evals,
                        n_bo_iter = n_bo_iter,
                        trial = trial,
                        n_first_batch = n_first_batch,
                        n_second_batch = n_second_batch
                )

