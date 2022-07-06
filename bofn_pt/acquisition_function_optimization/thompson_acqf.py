from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils.gp_sampling import GPDraw
import torch

class ThompsonSampling:
    def __init__(
        self,
        first_layer_GPs,
        random_seeds_first_layer,
        second_layer_GP,
        random_seed_second_layer,
        n_first_layer_nodes,
        normal_lower,
        normal_upper
    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            first_layer_GPs: Fitted GP models for functions in the first layer node
            second_layer_GPs: A Fitted GP model for the function in the second layer node
        """
        self.first_layer_GPs = first_layer_GPs
        self.second_layer_GP = second_layer_GP
        self.random_seeds_first_layer = random_seeds_first_layer
        self.random_seed_second_layer = random_seed_second_layer
        self.n_first_layer_nodes = n_first_layer_nodes
        self.normal_lower = normal_lower
        self.normal_upper =normal_upper
        self.thompson_first_layer = [None for i in range(self.n_first_layer_nodes)]
        for i in range(self.n_first_layer_nodes):
            self.thompson_first_layer[i] = GPDraw(model=self.first_layer_GPs[i],seed=self.random_seeds_first_layer[i])
        self.thompson_second_layer = GPDraw(model=self.second_layer_GP,seed=self.random_seed_second_layer)

    def forward(self, X: Tensor) -> Tensor:
        ts_vals_first_layer = torch.tensor([])
        for i in range(self.n_first_layer_nodes):
            ts_vals_first_layer= torch.cat((ts_vals_first_layer,self.thompson_first_layer[i].forward(X=X).detach()),1)
        ts_vals_first_layer_norm = ts_vals_first_layer.clone()
        for i in range(self.n_first_batch):
            ts_vals_first_layer_norm =(ts_vals_first_layer_norm[..., i] - normal_lower[i]) / (
                        normal_upper[i] - normal_lower[i])
        ts_val = self.thompson_second_layer.forward(X=ts_vals_first_layer_norm)
        return ts_val