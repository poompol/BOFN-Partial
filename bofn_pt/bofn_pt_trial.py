import numpy as np
import os
import sys
import time
import torch
import pygad
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.acquisition import PosteriorMean as GPPosteriorMean
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
from bofn_pt.acquisition_function_optimization.thompson_acqf import ThompsonSampling
from botorch.models.transforms import Standardize

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
        results_folder = script_dir + "/results/" + problem + "/" + first_batch_algo + "_" + second_batch_algo +"/"
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
        print("First Batch algo: " + first_batch_algo)
        print("Second Batch algo: "+ second_batch_algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        t0 = time.time()
        objective_at_X=objective_at_X.reshape((len(objective_at_X),1))

        if first_batch_algo== None and second_batch_algo == "RAND":
            second_batch = torch.rand([n_second_batch, input_dim])
        elif first_batch_algo== None and second_batch_algo == "TS-Whole":
            second_batch = generate_batch_thompson(first_layer_input=first_layer_input,
            first_layer_output=first_layer_output,second_layer_input=first_layer_output,
            second_layer_output=objective_at_X,batch_size=n_second_batch,n_candidates=500,
            network_to_objective_transform=network_to_objective_transform)
            y_at_new_point = function_network(second_batch)[:, :-1].clone()
            first_layer_input = torch.cat((first_layer_input,second_batch),0)
            first_layer_output = torch.cat((first_layer_output,y_at_new_point),0)
        elif first_batch_algo in ["EIFN-NoCL","EIFN-CLMAX","EIFN-CLMIN","EIFN-CLMEAN"]:
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
                if first_batch_algo == "EIFN-CLMAX":
                    CL = objective_at_X.max()
                    second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                    second_layer_output = torch.cat((second_layer_output,CL.reshape(1,1)),0)
                elif first_batch_algo == "EIFN-CLMIN":
                    CL = objective_at_X.min()
                    second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                    second_layer_output = torch.cat((second_layer_output,CL.reshape(1,1)),0)
                elif first_batch_algo == "EIFN-CLMEAN":
                    CL = model_second_layer.posterior(y_at_new_point_norm).mean.item()
                    second_layer_input = torch.cat((second_layer_input,y_at_new_point),0)
                    second_layer_output = torch.cat((second_layer_output,torch.tensor(CL).reshape(1,1)),0)
            second_batch = get_second_batch(first_batch = first_batch, temp_second_layer_input=temp_second_layer_input,
            model_second_layer=model_second_layer, best_obs_val = best_obs_val,
            n_second_batch = n_second_batch,second_batch_algo=second_batch_algo)
        elif first_batch_algo in ["TS-NoCL", "TS-CLMAX", "TS-CLMIN", "TS-CLMEAN"]:
            print("To be filled")
        else:
            print(f"Invalid algorithms")

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

    # Model
    model = GaussianProcessNetworkPartial(train_X = first_layer_input,
                                          train_Y_output_first_layer = first_layer_output,
                                          train_Y_input_second_layer = second_layer_input,
                                          train_Z = second_layer_output)
    qmc_sample = SobolQMCNormalSampler(num_samples=128)
    if first_batch_algo in ["EIFN-NoCL","EIFN-CLMAX","EIFN-CLMIN","EIFN-CLMEAN"]:
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
            posterior_mean = posterior_mean_function
        )
    elif first_batch_algo in ["TS-NoCL", "TS-CLMAX", "TS-CLMIN", "TS-CLMEAN"]:
        acquisition_function =

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
    elif second_batch_algo == "TS":
        print("To be filled")

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
    return new_second_batch

def generate_batch_thompson(
    first_layer_input: Tensor,
    first_layer_output: Tensor,
    second_layer_input: Tensor,
    second_layer_output: Tensor,
    batch_size,
    n_candidates,
    network_to_objective_transform,
    sampler="rff",  # "cholesky", "ciq", "rff"
    use_keops=False,
):
    assert sampler in ("cholesky", "ciq", "rff", "lanczos")
    assert first_layer_input.min() >= 0.0 and first_layer_input.max() <= 1.0 and torch.all(torch.isfinite(second_layer_output))
    # Fit a GP model
    model = GaussianProcessNetworkPartial(train_X = first_layer_input,
    train_Y_output_first_layer = first_layer_output,
    train_Y_input_second_layer = second_layer_input,
    train_Z = second_layer_output)
    # Draw samples on a Sobol sequence
    sobol = SobolEngine(first_layer_input.shape[-1], scramble=True)
    X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    # Thompson sample
    with ExitStack() as es:
        if sampler == "cholesky":
            es.enter_context(gpts.max_cholesky_size(float("inf")))
        elif sampler == "ciq":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(True))
            es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
            es.enter_context(gpts.num_contour_quadrature(15))
        elif sampler == "lanczos":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            es.enter_context(gpts.max_cholesky_size(0))
            es.enter_context(gpts.ciq_samples(False))
        elif sampler == "rff":
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False,objective = network_to_objective_transform)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
    return X_next
