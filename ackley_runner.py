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
from testcases.ackley import Ackley
ackley = Ackley()
input_dim = ackley.input_dim
n_layers = 2
problem = 'ackley'

def function_network(X: Tensor):
    return ackley.evaluate(X=X)

# Function that maps the network output to the objective value
# Extract only the final column of a tensor which corresponds to objective function values
network_to_objective_transform = lambda Y:Y[...,-1]
network_to_objective_transform = GenericMCObjective(network_to_objective_transform)

# Run experiment
'''
Specify algorithm:
Possible choices for "first_batch_algo": EIFN / TSFN / None
Possible choices for "Constant_Liar": NoCL / CLMAX / CLMIN / CLMEAN / None
Possible choices for "second_batch_algo" (This argument is required for every algorithm) : qEI / TS / RAND / TSWH
'''
first_batch_algo = "TSFN"
Constant_Liar = "CLMAX"
second_batch_algo = 'qEI'

# number of candidates in first and second batches, respectively. Second batch candidates are selected from the first one
n_first_batch = 3
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
    Constant_Liar = Constant_Liar,
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
