# Need two inputs from user which specify trial runs
import os
import sys
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.settings import debug
from torch import Tensor
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from bofn_pt.acquisition_function_optimization.thompsom_fn_acqf import ThompsonSampling
from bofn_pt.acquisition_function_optimization.optimize_acqf import optimize_acqf_and_get_suggested_point
neg_hartmann6 = Hartmann(dim=6, negate=True)
input_dim = 6
train_x = torch.rand(10, 6)
train_first_output = neg_hartmann6(train_x).unsqueeze(-1)
train_second_input = torch.tensor([])
n_first_layer_nodes=6
for i in range(n_first_layer_nodes):
    train_second_input=torch.cat((train_second_input,train_first_output.clone()),dim=1)
second_layer_output = neg_hartmann6(train_second_input).unsqueeze(-1)
model_first_layer = [None for i in range(n_first_layer_nodes)]
mll_first_layer = [None for i in range(n_first_layer_nodes)]
for i in range(6):
    model_first_layer[i]=SingleTaskGP(train_X=train_x, train_Y=train_first_output,
                               outcome_transform=Standardize(m=1))
    mll_first_layer[i] = ExactMarginalLogLikelihood(model_first_layer[i].likelihood, model_first_layer[i])
    fit_gpytorch_model(mll_first_layer[i])
normal_lower = [None for i in range(n_first_layer_nodes)]
normal_upper = [None for i in range(n_first_layer_nodes)]
train_second_input_norm = train_second_input.clone()
for i in range(n_first_layer_nodes):
    normal_lower[i] = torch.min(train_second_input[..., i])
    normal_upper[i] = torch.max(train_second_input[..., i])
    train_second_input_norm [..., i] = (train_second_input_norm [..., i] - normal_lower[i]) / (
            normal_upper[i] - normal_lower[i])
model_second_layer = SingleTaskGP(train_X=train_second_input_norm, train_Y=second_layer_output,
                               outcome_transform=Standardize(m=1))
acquisition_function = ThompsonSampling(first_layer_GPs=model_first_layer,second_layer_GP=model_second_layer,
                                        n_first_layer_nodes=n_first_layer_nodes,
                                        normal_lower=normal_lower,normal_upper=normal_upper)
test_x= torch.rand(2,6)
aa=acquisition_function.forward(X=test_x)
new_first_batch = optimize_acqf_and_get_suggested_point(
    acq_func=acquisition_function,
    bounds=torch.tensor([[0. for i in range(input_dim)],
                         [1. for i in range(input_dim)]]),
    batch_size=1,
    batch_limit=1,
    init_batch_limit=1
)