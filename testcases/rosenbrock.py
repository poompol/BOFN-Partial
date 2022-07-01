import torch
import numpy as np

class Rosenbrock:

    def __init__(self):
        self.n_layers = 2  # no of function layers
        self.input_dim = 5  # problem dimension
        self.n_nodes_each_layer = [8,1] # number of nodes in first layer


    def evaluate(self,X):
        X_scale = 4 * X -2
        input_shape = X_scale.shape
        n_col_output = np.sum(self.n_nodes_each_layer)
        output = torch.empty(input_shape[:-1] + torch.Size([n_col_output]))
        temp = torch.empty(input_shape[:-1] + torch.Size([n_col_output]))
        for in_node in range(self.n_nodes_each_layer[0]):
            if in_node < 4:
                output[..., in_node] = X_scale[:,in_node+1]-X_scale[:,in_node].pow(2)
                temp[..., in_node] = 100*output[..., in_node].pow(2)
            else:
                output[..., in_node] = X_scale[:, in_node-4]
                temp[..., in_node] =(1-X_scale[:, in_node-4]).pow(2)
        output[..., -1] = -torch.sum(temp[..., :-1],1)
        return output