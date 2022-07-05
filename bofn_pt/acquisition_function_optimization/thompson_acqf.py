from botorch.utils.gp_sampling import GPDraw
class thompsonSampling(MCAcquisitionFunction):
    r"""MC-based Thompson Sampling Acquisition Function for Function Network
    """

    def __init__(
        self,
        first_layer_GPs,
        random_seeds_first_layer,
        second_layer_GP,
        radnom_seed_second_layer,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs: Any,
    ) -> None:
        r"""Thompson Sampling Acquisition Function.

        Args:
            first_layer_GPs: Fitted GP models for functions in the first layer node
            second_layer_GPs: A Fitted GP model for the function in the second layer node
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=512, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        super().__init__(
            first_layer_GPs = first_layer_GPs,
            second_layer_GP = second_layer_GP
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
        )

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.sampler(posterior)
        obj = self.objective(samples, X=X)
        obj = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)
        q_ei = obj.max(dim=-1)[0].mean(dim=0)
        return q_ei