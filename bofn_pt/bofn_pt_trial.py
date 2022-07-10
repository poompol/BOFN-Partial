import numpy as np
import os
import sys
import time
import torch
import pygad
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement #(only qEI will be called)
#from botorch.acquisition import PosteriorMean as GPPosteriorMean (can be deleted, not be called)
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor
from typing import Callable, List, Optional

from bofn_pt.utils.generate_initial_design import generate_initial_design
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from bofn_pt.models.gp_network_partial import GaussianProcessNetworkPartial
from bofn_pt.utils.posterior_mean import PosteriorMean
from bofn_pt.acquisition_function_optimization.optimize_acqf import optimize_acqf_and_get_suggested_point
from bofn_pt.acquisition_function_optimization.thompson_fn_acqf import ThompsonSamplingFN
from bofn_pt.acquisition_function_optimization.thompson_2nd_acqf import ThompsonSampling2ND
from botorch.models.transforms import Standardize
from botorch.optim.initializers import initialize_q_batch_nonneg

import os
import time
from contextlib import ExitStack
from torch.quasirandom import SobolEngine
import gpytorch
import gpytorch.settings as gpts
from botorch.generation import MaxPosteriorSampling
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel

device = torch.device("cpu")
dtype = torch.double
def bofn_pt_trial(
        problem: str,
        first_batch_algo: str,
        Constant_Liar: str,
        second_batch_algo: str,
        trial: int,
        n_init_evals: int,
        n_bo_iter: int,
        function_network: Callable,
        input_dim: int,
        network_to_objective_transform: Callable,
        n_first_batch: int,
        n_second_batch: int
) -> None:
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    if first_batch_algo is not None:
            results_folder = script_dir + "/results/" + problem + "/" + first_batch_algo +"-"+ Constant_Liar + "_" + second_batch_algo + "/"
    else:
            results_folder = script_dir + "/results/" + problem + "/" + second_batch_algo + "/"

    # Initial evaluations
    X = generate_initial_design(
        num_samples=n_init_evals, input_dim=input_dim, seed=trial
    )
    network_output_at_X = function_network(X)
    objective_at_X = network_to_objective_transform(network_output_at_X)

    first_layer_input = X
    first_layer_output = network_output_at_X[:,:-1]
    n_first_layer_nodes = first_layer_output.shape[1]

    # Current Best Objective value
    best_obs_val = objective_at_X.max().item()

    # Historical best observed objective values and running times
    hist_best_obs_vals = [best_obs_val]
    runtimes = []

    init_batch_id = 1
    for iteration in range(init_batch_id, n_bo_iter+1):
        print("Experiment: " + problem)
        if first_batch_algo is not None:
            print("First Batch algo: " + first_batch_algo)
        if Constant_Liar is not None:
            print("Constant Liar: " + Constant_Liar)
        print("Second Batch algo: "+ second_batch_algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        t0 = time.time()
        objective_at_X=objective_at_X.reshape((len(objective_at_X),1))

        if first_batch_algo == None and Constant_Liar == None:
            if second_batch_algo == "RAND":
                second_batch = torch.rand([n_second_batch, input_dim])
            elif second_batch_algo == "TSWH":
                second_layer_input = network_output_at_X[:,:-1].clone()
                second_layer_output = objective_at_X.clone()
                second_batch = get_second_batch_TS_whole(first_layer_input=first_layer_input,
                first_layer_output=first_layer_output,second_layer_input=second_layer_input,
                second_layer_output=second_layer_output,n_first_layer_nodes=n_first_layer_nodes,n_second_batch=n_second_batch)
            else:
                print("Invalid algorithm")
                break
        else:
            second_layer_input= network_output_at_X[:,:-1].clone()
            second_layer_output = objective_at_X.clone()
            second_layer_input_norm = second_layer_input.clone()
            normal_lower = [None for i in range(n_first_layer_nodes)]
            normal_upper = [None for i in range(n_first_layer_nodes)]
            for i in range(n_first_layer_nodes):
                normal_lower[i]=torch.min(network_output_at_X[...,i])
                normal_upper[i]=torch.max(network_output_at_X[...,i])
                second_layer_input_norm[...,i] = (second_layer_input_norm[...,i]-normal_lower[i])/(normal_upper[i]-normal_lower[i])
            model_second_layer= SingleTaskGP(train_X=second_layer_input_norm, train_Y=objective_at_X,outcome_transform=Standardize(m=1))
            mll_second_layer = ExactMarginalLogLikelihood(model_second_layer.likelihood, model_second_layer)
            fit_gpytorch_model(mll_second_layer)
            for inner_iter in range(n_first_batch):
                print(f"Choosing first batch point: "+str(inner_iter+1))
                new_point = get_first_batch(first_layer_input=first_layer_input,
                                                first_layer_output=first_layer_output,
                                                second_layer_input=second_layer_input,
                                                second_layer_output=second_layer_output,
                                                network_to_objective_transform=network_to_objective_transform,
                                                n_first_layer_nodes=n_first_layer_nodes,
                                                first_batch_algo=first_batch_algo)
                y_at_new_point = function_network(new_point)[:, :-1].clone()
                first_layer_input = torch.cat((first_layer_input,new_point),0)
                first_layer_output = torch.cat((first_layer_output,y_at_new_point),0)
                y_at_new_point_norm = y_at_new_point.clone()
                for i in range(n_first_layer_nodes):
                    y_at_new_point_norm[...,i]=(y_at_new_point_norm[...,i]-normal_lower[i])/(normal_upper[i]-normal_lower[i])
                if inner_iter == 0:
                    first_batch = new_point
                    temp_second_layer_input = y_at_new_point_norm
                else:
                    first_batch = torch.cat((first_batch,new_point),0)
                    temp_second_layer_input = torch.cat((temp_second_layer_input,y_at_new_point_norm),0)
                if Constant_Liar != "NoCL":
                    if Constant_Liar == "CLMAX":
                        CL = objective_at_X.max()
                        second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                        second_layer_output = torch.cat((second_layer_output,CL.reshape(1,1)),0)
                    elif Constant_Liar == "CLMIN":
                        CL = objective_at_X.min()
                        second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                        second_layer_output = torch.cat((second_layer_output,CL.reshape(1,1)),0)
                    elif Constant_Liar == "CLMEAN":
                        CL = model_second_layer.posterior(y_at_new_point_norm).mean.item()
                        second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                        second_layer_output = torch.cat((second_layer_output,torch.tensor(CL).reshape(1,1)),0)
            print(temp_second_layer_input)
            print(temp_second_layer_input.shape)
            second_batch = get_second_batch(first_batch = first_batch, temp_second_layer_input=temp_second_layer_input,
            model_second_layer=model_second_layer, best_obs_val = best_obs_val,
            n_second_batch = n_second_batch,second_batch_algo=second_batch_algo)
        t1 = time.time()
        runtimes.append(t1-t0)

        # Evaluate at new (second) batch
        network_output_at_new_batch = function_network(second_batch)
        objective_at_new_batch = network_to_objective_transform(network_output_at_new_batch)
        objective_at_new_batch = objective_at_new_batch.reshape((objective_at_new_batch.shape[0],1))

        #Update training data
        X= torch.cat([X,second_batch],0)
        network_output_at_X = torch.cat([network_output_at_X,network_output_at_new_batch],0)
        objective_at_X = torch.cat([objective_at_X,objective_at_new_batch],0)

        #Update historical best observed objective values
        best_obs_val = objective_at_X.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))

        # Save data
        np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        np.savetxt(results_folder + "network_output_at_X/network_output_at_X_"+str(trial)+".txt",network_output_at_X.numpy())
        np.savetxt(results_folder + "objective_at_X/objective_at_X_" + str(trial)+".txt",objective_at_X.numpy())
        np.savetxt(results_folder + "best_obs_vals_"+str(trial)+".txt",np.atleast_1d(hist_best_obs_vals))
        np.savetxt(results_folder + "runtimes/runtimes_"+str(trial)+".txt",np.atleast_1d(runtimes))

def get_first_batch(
        first_layer_input: Tensor,
        first_layer_output: Tensor,
        second_layer_input: Tensor,
        second_layer_output: Tensor,
        first_batch_algo: str,
        n_first_layer_nodes: int,
        network_to_objective_transform: Callable
)-> Tensor:
    input_dim = first_layer_input.shape[-1]
    if first_batch_algo == "EIFN":
        model = GaussianProcessNetworkPartial(train_X = first_layer_input,
                                        train_Y_output_first_layer = first_layer_output,
                                        train_Y_input_second_layer = second_layer_input,
                                        train_Z = second_layer_output)
        qmc_sample = SobolQMCNormalSampler(num_samples=128)
        acquisition_function = qExpectedImprovement(
            model = model,
            best_f = second_layer_output.max().item(),
            sampler = qmc_sample,
            objective = network_to_objective_transform
        )
        posterior_mean_function = PosteriorMean(
            model = model,
            sampler = qmc_sample,
            objective = network_to_objective_transform
        )
        new_first_batch = optimize_acqf_and_get_suggested_point(
            acq_func = acquisition_function,
            bounds=torch.tensor([[0. for i in range(input_dim)],
                                 [1. for i in range(input_dim)]]),
            batch_size=1,
            posterior_mean = posterior_mean_function,
        )
    elif first_batch_algo == "TSFN":
        first_layer_GPs = [None for i in range(n_first_layer_nodes)]
        for i in range(n_first_layer_nodes):
            first_layer_GPs[i] = SingleTaskGP(train_X=first_layer_input,train_Y=first_layer_output[...,[i]],outcome_transform=Standardize(m=1))
        second_layer_input_norm = second_layer_input.clone()
        normal_lower = [None for i in range(n_first_layer_nodes)]
        normal_upper = [None for i in range(n_first_layer_nodes)]
        for i in range(n_first_layer_nodes):
            normal_lower[i] = torch.min(second_layer_input[..., i])
            normal_upper[i] = torch.max(second_layer_input[..., i])
            second_layer_input_norm[..., i] = (second_layer_input_norm[..., i] - normal_lower[i]) / (
                        normal_upper[i] - normal_lower[i])
        second_layer_GP  = SingleTaskGP(train_X=second_layer_input_norm, train_Y=second_layer_output,
                                          outcome_transform=Standardize(m=1))
        acquisition_function = ThompsonSamplingFN(first_layer_GPs=first_layer_GPs,
                                                second_layer_GP=second_layer_GP,
                                                n_first_layer_nodes=n_first_layer_nodes,
                                                normal_lower=normal_lower,normal_upper=normal_upper)
        new_first_batch = optimize_acqf_and_get_suggested_point(
            acq_func = acquisition_function,
            bounds=torch.tensor([[0. for i in range(input_dim)],
                                 [1. for i in range(input_dim)]]),
            batch_size=1,
            batch_limit=1,
            init_batch_limit=1
        )
    else:
        print("Invalid algorithm!")
    return new_first_batch

def get_second_batch(first_batch: Tensor,temp_second_layer_input: Tensor,model_second_layer,
                     best_obs_val: float,n_second_batch: int, second_batch_algo: str)-> Tensor:
    if second_batch_algo == "qEI":
        sampler = SobolQMCNormalSampler(num_samples=128)
        qEI = qExpectedImprovement(model_second_layer, best_obs_val, sampler)
        n_first_batch = first_batch.shape[0]
        gene_space = [list(range(n_first_batch)) for i in range(n_second_batch)]
        def fitness_func(X,solution_idx):
            selected_tensor = temp_second_layer_input[X,:]
            fitness_val = qEI(selected_tensor).item()
            return fitness_val
        ga_instance = pygad.GA(num_generations=50,
                           sol_per_pop=10,
                           num_genes=n_second_batch,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           gene_type=int,
                           gene_space=gene_space,
                           allow_duplicate_genes=False)
        ga_instance.run()
        selected_indices = ga_instance.best_solution()[0]
        new_second_batch = first_batch[selected_indices,:].clone()
    elif second_batch_algo == "TS":
        new_second_batch = torch.tensor([])
        for i in range(n_second_batch):
            acquisition_function = ThompsonSampling2ND(second_layer_GP=model_second_layer)
            thompson_val = acquisition_function.forward(X=temp_second_layer_input)
            arg_max_index = torch.argmax(thompson_val)
            if i==0:
                new_second_batch = first_batch[arg_max_index,:].clone().reshape(1,first_batch.shape[-1])
            else:
                new_second_batch = torch.cat((new_second_batch,
                first_batch[arg_max_index,:].clone().reshape(1,first_batch.shape[-1])),dim=0)
            temp_second_layer_input=torch.cat((temp_second_layer_input[:arg_max_index],
            temp_second_layer_input[arg_max_index+1:]))
            first_batch = torch.cat((first_batch[:arg_max_index],
            first_batch[arg_max_index+1:]))
    return new_second_batch

def get_second_batch_TS_whole(
    first_layer_input: Tensor,
    first_layer_output: Tensor,
    n_first_layer_nodes: int,
    second_layer_input: Tensor,
    second_layer_output: Tensor,
    n_second_batch: int,
):
    input_dim = first_layer_input.shape[-1]
    first_layer_GPs = [None for i in range(n_first_layer_nodes)]
    for i in range(n_first_layer_nodes):
        first_layer_GPs[i] = SingleTaskGP(train_X=first_layer_input,train_Y=first_layer_output[...,[i]],outcome_transform=Standardize(m=1))
    second_layer_input_norm = second_layer_input.clone()
    normal_lower = [None for i in range(n_first_layer_nodes)]
    normal_upper = [None for i in range(n_first_layer_nodes)]
    for i in range(n_first_layer_nodes):
        normal_lower[i] = torch.min(second_layer_input[..., i])
        normal_upper[i] = torch.max(second_layer_input[..., i])
        second_layer_input_norm[..., i] = (second_layer_input_norm[..., i] - normal_lower[i]) / (
                    normal_upper[i] - normal_lower[i])
    second_layer_GP  = SingleTaskGP(train_X=second_layer_input_norm, train_Y=second_layer_output,
                                        outcome_transform=Standardize(m=1))
    for i in range(n_second_batch):
        acquisition_function = ThompsonSamplingFN(first_layer_GPs=first_layer_GPs,
                                                second_layer_GP=second_layer_GP,
                                                n_first_layer_nodes=n_first_layer_nodes,
                                                normal_lower=normal_lower,normal_upper=normal_upper)
        test_x = torch.rand(1,input_dim)
        new_point = optimize_acqf_and_get_suggested_point(
            acq_func = acquisition_function,
            bounds=torch.tensor([[0. for i in range(input_dim)],
                                [1. for i in range(input_dim)]]),
            batch_size=1,
            batch_limit=1,
            init_batch_limit=1
        )
        if i==0:
            new_second_batch = new_point
        else:
            new_second_batch = torch.cat((new_second_batch,new_point),dim=0)
        print("this is second batch")
        print(new_second_batch)
    return new_second_batch
