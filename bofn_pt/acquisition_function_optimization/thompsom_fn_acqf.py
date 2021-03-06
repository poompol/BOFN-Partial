import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.utils.gp_sampling import get_gp_samples
from typing import Callable, List, Optional
from botorch.models.model import Model
class ThompsonSamplingFN(AcquisitionFunction):
    def __init__(
        self,
        first_layer_GPs: List[Model],
        second_layer_GP: Model,
        n_first_layer_nodes,
        normal_lower,
        normal_upper
    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            first_layer_GPs: A List of Fitted GP models for functions in the first layer node
            second_layer_GPs: A Fitted GP model for the function in the second layer node
        """
        super(AcquisitionFunction, self).__init__()
        self.first_layer_GPs = first_layer_GPs
        self.second_layer_GP = second_layer_GP
        self.n_first_layer_nodes = n_first_layer_nodes
        self.normal_lower = normal_lower
        self.normal_upper =normal_upper
        self.thompson_first_layer = [None for i in range(self.n_first_layer_nodes)]
        for i in range(self.n_first_layer_nodes):
            self.thompson_first_layer[i] = get_gp_samples(model=self.first_layer_GPs[i], num_outputs=1, n_samples=1)
        self.thompson_second_layer = get_gp_samples(model=self.second_layer_GP, num_outputs=1, n_samples=1)

    def forward(self, X: Tensor) -> Tensor:
        ts_vals_first_layer = torch.tensor([])
        for i in range(self.n_first_layer_nodes):
            ts_vals_first_layer= torch.cat((ts_vals_first_layer,self.thompson_first_layer[i].forward(X=X)),dim=-1)
        ts_vals_first_layer_norm = ts_vals_first_layer.clone()
        for i in range(self.n_first_layer_nodes):
            ts_vals_first_layer_norm[...,i] =(ts_vals_first_layer_norm[..., i] - self.normal_lower[i]) / (
                        self.normal_upper[i] - self.normal_lower[i])
        ts_val = self.thompson_second_layer.forward(X=ts_vals_first_layer_norm).squeeze(-1).squeeze(0)
        return ts_val
