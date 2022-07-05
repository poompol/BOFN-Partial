# Need two inputs from user which specify trial runs
import os
import sys
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor


torch.set_default_dtype(torch.float64)
debug._set_state(True)

from bofn_pt.experiment_manager import experiment_manager

# Objective function (network)
from testcases.rosenbrock import Rosenbrock
rosenbrock = Rosenbrock()
input_dim = rosenbrock.input_dim
n_layers = 2
problem = 'rosenbrock'

def function_network(X: Tensor):
    return rosenbrock.evaluate(X=X)

# Function that maps the network output to the objective value
# Extract only the final column of a tensor which corresponds to objective function values
network_to_objective_transform = lambda Y:Y[...,-1]
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

# Run experiment
# First Batch Algo > Current possible choices: EIFN-NoCL/EIFN-CLMAX/EIFN-CLMIN/EIFN-CLMEAN/
# TS-NoCL/TS-CLMAX/TS-CLMIN/TS-CLMEAN/ None (for RAND and TS-Whole)
first_batch_algo = 'EIFN-NoCL'
# Second Batch Algo > Current possible choices: qEI/TS/RAND/TS-Whole
second_batch_algo = 'qEI'

# number of candidates in first and second batches, respectively. Second batch candidates are selected from the first one
n_first_batch = 10
n_second_batch = 2


n_bo_iter = 20 # no of maximum no of iterations for BO main loops

# no of replications (inputs from system)
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])


experiment_manager(
    problem = problem,
    first_batch_algo = first_batch_algo,
    second_batch_algo = second_batch_algo,
    first_trial = first_trial,
    last_trial = last_trial,
    n_init_evals = 2 *(input_dim+1),
    n_bo_iter = n_bo_iter,
    function_network = function_network,
    input_dim = input_dim,
    network_to_objective_transform = network_to_objective_transform,
    n_first_batch = n_first_batch,
    n_second_batch = n_second_batch
)
