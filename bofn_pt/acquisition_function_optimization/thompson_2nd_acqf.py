import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction
from botorch.utils.gp_sampling import get_gp_samples
from typing import Callable, List, Optional
from botorch.models.model import Model
class ThompsonSampling2ND(AcquisitionFunction):
    def __init__(
        self,
        second_layer_GP: Model,

    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            second_layer_GPs: A Fitted GP model for the function in the second layer node
        """
        super(AcquisitionFunction, self).__init__()
        self.second_layer_GP = second_layer_GP
        self.thompson_second_layer = get_gp_samples(model=self.second_layer_GP, num_outputs=1, n_samples=1)

    def forward(self, X: Tensor) -> Tensor:
        ts_val = self.thompson_second_layer.forward(X=X).squeeze(-1).squeeze(0)
        return ts_val
