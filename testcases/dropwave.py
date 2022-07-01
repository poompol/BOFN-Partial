import torch

class Dropwave:

    def __init__(self):
        self.n_layers = 2 # no of function layers
        self.input_dim = 2 # problem dimension

    def evaluate(self,X):
        X_scale = 10.24 * X - 5.12
        input_shape = X_scale.shape
        output = torch.empty(input_shape[:-1]+torch.Size([self.n_layers])) # create empty tensor with dimension no obs x no layers
        norm_X = torch.norm(X_scale,dim = -1) # compute norm dim=0 >> by columns dim = 1/-1 >> by rows
        output[...,0] = norm_X
        output[...,1] = (1.0 + torch.cos(12.0 * norm_X))/(2.0 +0.5 * (norm_X ** 2))
        # out[...,0] is result of first layer and output[...,1] is result of final layer
        return output