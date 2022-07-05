from botorch.utils.gp_sampling import GPDraw
import torch

class ThompsonSampling:
    r"""MC-based Thompson Sampling Acquisition Function for Function Network
    """

    def __init__(
        self,
        first_layer_GPs,
        random_seeds_first_layer,
        second_layer_GP,
        random_seed_second_layer,
        n_first_layer_nodes,
        second_layer_function,
    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            first_layer_GPs: Fitted GP models for functions in the first layer node
            second_layer_GPs: A Fitted GP model for the function in the second layer node
        """
        self.first_layer_GPs = first_layer_GPs,
        self.second_layer_GP = second_layer_GP,
        self.random_seeds_first_layer = random_seeds_first_layer,
        self.random_seed_second_layer = random_seed_second_layer,
        self.n_first_layer_nodes = n_first_layer_nodes,
        self.thompson_first_layer = [None for i in range(self.n_first_layer_nodes)]
        for i in range(self.n_first_layer_nodes):
            self.thompson_first_layer[i] = GPDraw(model=self.first_layer_GPs[i],seed=self.random_seeds_first_layer[i])
        self.thompson_second_layer = GPDraw(model=self.second_layer_GP,seed=self.random_seed_seed_layer)

    def forward(self, X: Tensor) -> Tensor:
        ts_vals_first_layer = torch.tensor([])
        for i in range(self.n_first_layer_nodes):
            ts_vals_first_layer= torch.cat((ts_vals_first_layer,self.thompson_first_layer[i].forward(X=X).detach()),1)
        ts_val_second_layer = self.thompson_second_layer.forward(X=ts_vals_first_layer)
        return ts_val_second_layer