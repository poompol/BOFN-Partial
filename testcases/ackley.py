import torch
import numpy as np

class Ackley:

    def __init__(self):
        self.n_layers = 2  # no of function layers
        self.input_dim = 6  # problem dimension
        self.n_nodes_each_layer = [2,1] # number of nodes in first layer

    def evaluate(self,X):
        X_scale = 4 * X -2
        input_shape = X_scale.shape
        n_col_output = np.sum(self.n_nodes_each_layer)
        output = torch.empty(input_shape[:-1] + torch.Size([n_col_output]))
        output[...,0]= (1/input_shape[1])*torch.sum(X_scale.pow(2),dim=-1)
        output[...,1]= (1 / input_shape[1]) * torch.sum(torch.cos(2 * torch.pi * X_scale),dim=-1)
        output[...,2]= 20 * torch.exp(-0.2*torch.sqrt(output[...,0]))+torch.exp(output[...,1])-20-torch.exp(torch.tensor([1]))
        # out[...,0] and out[...,2] are results of first layer and output[...,1] is result of final layer
        return output
