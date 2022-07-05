#! /usr/bin/env python3

r"""
Gaussian Process Network Partial
"""

from __future__ import annotations
import torch
from typing import Any
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from botorch.models.transforms import Standardize


class GaussianProcessNetworkPartial(Model):
    def __init__(self, train_X, train_Y_output_first_layer, train_Y_input_second_layer, train_Z,first_layer_GPs=None, second_layer_GP=None,normalization_constant_upper=None, normalization_constant_lower=None) -> None:
        super(Model, self).__init__()
        self.train_X = train_X
        self.train_Y_output_first_layer = train_Y_output_first_layer
        self.train_Y_input_second_layer = train_Y_input_second_layer
        self.train_Z = train_Z
        self.n_first_layer_nodes = train_Y_output_first_layer.shape[1]
        if first_layer_GPs is not None:
            self.first_layer_GPs = first_layer_GPs
        else:
            self.first_layer_GPs = [None for k in range(self.n_first_layer_nodes)]
            self.first_layer_mlls = [None for k in range(self.n_first_layer_nodes)]
            for k in range(self.n_first_layer_nodes):
                train_X_first_layer_node_k = train_X
                train_Y_first_layer_k = train_Y_output_first_layer[..., [k]]
                self.first_layer_GPs[k] = SingleTaskGP(train_X=train_X_first_layer_node_k, train_Y=train_Y_first_layer_k,outcome_transform=Standardize(m=1))
                self.first_layer_mlls[k] = ExactMarginalLogLikelihood(self.first_layer_GPs[k].likelihood,self.first_layer_GPs[k])
                fit_gpytorch_model(self.first_layer_mlls[k])
        if second_layer_GP is not None:
            self.second_layer_GP = second_layer_GP
            self.normalization_constant_upper = normalization_constant_upper
            self.normalization_constant_lower =normalization_constant_lower
        else:
            self.normalization_constant_upper=[None for i in range(self.n_first_layer_nodes)]
            self.normalization_constant_lower=[None for i in range(self.n_first_layer_nodes)]
            aux = train_Y_input_second_layer.clone()
            for i in range(self.n_first_layer_nodes):
                self.normalization_constant_upper[i] = torch.max(aux[...,i])
                self.normalization_constant_lower[i] = torch.min(aux[...,i])
                aux[...,i] = (aux[..., i] - self.normalization_constant_lower[i])/(self.normalization_constant_upper[i] - self.normalization_constant_lower[i])
            train_Y_input_second_layer_norm = aux
            self.second_layer_GP = SingleTaskGP(train_X=train_Y_input_second_layer_norm, train_Y=train_Z,outcome_transform=Standardize(m=1))
            self.second_layer_mll = ExactMarginalLogLikelihood(self.second_layer_GP.likelihood, self.second_layer_GP)
            fit_gpytorch_model(self.second_layer_mll)


    def posterior(self, X: Tensor, observation_noise=False) -> MultivariateNormalNetworkPartial:
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return MultivariateNormalNetworkPartial(self.n_first_layer_nodes,self.first_layer_GPs, self.second_layer_GP, X,self.normalization_constant_lower, self.normalization_constant_upper)

    def forward(self, x: Tensor) -> MultivariateNormalNetworkPartial:
        return MultivariateNormalNetworkPartial(self.n_first_layer_nodes,self.first_layer_GPs, self.second_layer_GP, x,self.normalization_constant_lower, self.normalization_constant_upper)

    def condition_on_observations(self, X_new: Tensor, Y_output_first_layer_new: Tensor, Y_input_second_layer_new: Tensor or None, Z_new: Tensor or None, **kwargs: Any) -> Model:
        r"""Condition the model on new observations.
        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            Y: A `batch_shape' x n' x m`-dim Tensor, where `m` is the number of
                model outputs, `n'` is the number of points per batch, and
                `batch_shape'` is the batch shape of the observations.
                `batch_shape'` must be broadcastable to `batch_shape` using
                standard broadcasting semantics. If `Y` has fewer batch dimensions
                than `X`, it is assumed that the missing batch dimensions are
                the same for all `Y`.
        Returns:
            A `Model` object of the same type, representing the original model
            conditioned on the new observations `(X, Y)` (and possibly noise
            observations passed in via kwargs).
        """
        fantasy_first_layer_models = [None for k in range(self.n_first_layer_nodes)]

        for k in range(self.n_first_layer_nodes):
            X_first_layer_node_k = X_new
            Y_first_layer_node_k = Y_output_first_layer_new[..., [k]]
            fantasy_first_layer_models[k] = self.first_layer_GPs[k].condition_on_observations(X_first_layer_node_k,Y_first_layer_node_k)
        if Y_input_second_layer_new is not None and Z_new is not None:
            aux = Y_input_second_layer_new.clone()
            for i in range(self.n_first_layer_nodes):
                aux[...,i] = (aux[..., i] - self.normalization_constant_lower[i])/(self.normalization_constant_upper[i] - self.normalization_constant_lower[i])
            Y_input_second_layer_new_norm = aux
            fantasy_second_layer_model = self.second_layer_GP.condition_on_observations(Y_input_second_layer_new_norm,Z_new)
        else:
            fantasy_second_layer_model = self.second_layer_GP


        return GaussianProcessNetworkPartial(train_X=X_new, train_Y_output_first_layer=Y_output_first_layer_new,
                                             train_Y_input_second_layer=Y_input_second_layer_new, train_Z=Z_new,
                                             first_layer_GPs=fantasy_first_layer_models,
                                             second_layer_GP=fantasy_second_layer_model,
                                             normalization_constant_lower=self.normalization_constant_lower, 
                                             normalization_constant_upper=self.normalization_constant_upper)


class MultivariateNormalNetworkPartial(Posterior):
    def __init__(self, n_first_layer_nodes,first_layer_GPs, second_layer_GP, X,normalization_constant_lower=None,normalization_constant_upper=None):
        self.first_layer_GPs = first_layer_GPs
        self.second_layer_GP = second_layer_GP
        self.X = X
        self.n_first_layer_nodes = n_first_layer_nodes
        self.normalization_constant_lower = normalization_constant_lower
        self.normalization_constant_upper = normalization_constant_upper

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = self.n_first_layer_nodes + 1
        shape = torch.Size(shape)
        return shape

    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        for k in range(self.n_first_layer_nodes):
            X_node_k = self.X
            multivariate_normal_at_node_k = self.first_layer_GPs[k].posterior(X_node_k)
            if base_samples is not None:
                nodes_samples[..., k] = \
                multivariate_normal_at_node_k.rsample(sample_shape, base_samples=base_samples[..., [k]])[..., 0]
            else:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape)[..., 0]
        X_second_layer = nodes_samples[...,:-1]
        X_second_layer_norm = X_second_layer.clone()
        for i in range(self.n_first_layer_nodes):
            X_second_layer_norm[...,i] = (X_second_layer_norm[...,i]- self.normalization_constant_lower[i])/(self.normalization_constant_upper[i] - self.normalization_constant_lower[i])
        multivariate_normal_at_second_layer = self.second_layer_GP.posterior(X_second_layer_norm)
        if base_samples is not None:
            my_aux = torch.sqrt(multivariate_normal_at_second_layer.variance)
            if my_aux.ndim == 4:
                nodes_samples[...,self.n_first_layer_nodes] = (multivariate_normal_at_second_layer.mean + torch.einsum('abcd,a->abcd', torch.sqrt(multivariate_normal_at_second_layer.variance), torch.flatten(base_samples[..., self.n_first_layer_nodes])))[..., 0]
            elif my_aux.ndim == 5:
                nodes_samples[...,self.n_first_layer_nodes] = (multivariate_normal_at_second_layer.mean + torch.einsum('abcde,a->abcde', torch.sqrt(multivariate_normal_at_second_layer.variance), torch.flatten(base_samples[..., self.n_first_layer_nodes])))[..., 0]
            else:
                print(error)
        else:
            nodes_samples[...,self.n_first_layer_nodes]=multivariate_normal_at_second_layer.rsample()[0,..., 0]
        return nodes_samples
