"""
Gaussian Mixture Model utilities for 1D toy example.

Adapted from enhancedsampling repo for use with bioemu framework.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from bioemu.steering import CollectiveVariable

MU1 = -2.0
MU2 = 3.0
SIGMA1 = 0.5
SIGMA2 = 0.5


class TimeDependentGMM1D(nn.Module):
    """
    Time-dependent 1D Gaussian Mixture Model.

    The data distribution at t=0 is a GMM with two modes.
    At time t, the distribution evolves according to the forward SDE.
    """

    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        sigma1: float = SIGMA1,
        sigma2: float = SIGMA2,
        weight1: float = 0.7,
        scheduler=None,  # SDE scheduler (e.g., CosineVPSDE)
    ):
        super().__init__()
        self.register_buffer("mu1", mu1)  # [1]
        self.register_buffer("mu2", mu2)  # [1]
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.weight1 = weight1
        self.weight2 = 1.0 - weight1

        # Use provided scheduler (should be same as diffusion process)
        self.scheduler = scheduler

        # Log weights for numerical stability
        self.register_buffer("log_w1", torch.log(torch.tensor(weight1)))
        self.register_buffer("log_w2", torch.log(torch.tensor(1.0 - weight1)))

    def log_prob(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability at time t.

        Args:
            x: [batch_size, 1] - positions
            t: [batch_size] or scalar - time

        Returns:
            log_prob: [batch_size] - log probabilities
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        batch_size = x.shape[0]

        # Handle time broadcasting
        # if not isinstance(t, torch.Tensor):
        #     t = torch.tensor(t, device=x.device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        # Get marginal parameters at time t
        if hasattr(self.scheduler, "marginal_prob"):
            # CosineVPSDE or other SDE with marginal_prob method
            seq_length = 1
            dim = 1
            ones = torch.ones(batch_size, seq_length, dim, device=x.device)
            alpha_t, sigma_t = self.scheduler.marginal_prob(x=ones, t=t)
            alpha_t = alpha_t.squeeze()  # [batch_size]
            sigma_t = sigma_t.squeeze()  # [batch_size]
        else:
            # BetaSchedule fallback
            alpha_t, sigma_t = self.scheduler.get_alpha_t_sigma_t(t)  # [batch_size]

        # Compute marginal means and stds for each component
        # mu_t = alpha_t * mu_0
        mu1_t = alpha_t.unsqueeze(-1) * self.mu1  # [batch_size, 1]
        mu2_t = alpha_t.unsqueeze(-1) * self.mu2  # [batch_size, 1]

        # sigma_t^2 = sigma_0^2 * alpha_t^2 + sigma_noise^2
        sigma1_t = torch.sqrt(self.sigma1**2 * alpha_t**2 + sigma_t**2).unsqueeze(
            -1
        )  # [batch_size, 1]
        sigma2_t = torch.sqrt(self.sigma2**2 * alpha_t**2 + sigma_t**2).unsqueeze(
            -1
        )  # [batch_size, 1]

        # Compute log probabilities for each component
        log_prob1 = Normal(mu1_t, sigma1_t).log_prob(x).sum(dim=-1)  # [batch_size]
        log_prob2 = Normal(mu2_t, sigma2_t).log_prob(x).sum(dim=-1)  # [batch_size]

        # Use log-sum-exp trick for mixture
        log_probs = torch.stack(
            [log_prob1 + self.log_w1, log_prob2 + self.log_w2], dim=-1
        )  # [batch_size, 2]

        return torch.logsumexp(log_probs, dim=-1)  # [batch_size]

    @torch.enable_grad()
    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute score function using autograd: ∇_x log q(x_t).

        By default this is a first-order quantity used for sampling only, so
        the computation graph is not kept. When ``create_graph=True``, the
        returned score retains its computation graph so that gradients of any
        downstream quantity that depends on the score with respect to ``x`` or
        ``t`` can be computed (i.e. enables second-order derivatives).
        """

        if create_graph:
            # Use x directly (or a cloned version) without detaching so that
            # the graph from log_prob to the original x is preserved.
            x_var = x

        else:
            # Default: no higher-order derivatives required. Detach x to keep
            # this computation inexpensive and side-effect free for callers that
            # only need the score value.
            x_var = x.clone().requires_grad_(True)

        # Compute log probability of marginal distribution at time t
        log_prob = self.log_prob(x_var, t)

        # Sum log probabilities to get scalar for autograd (needed for gradient computation)
        if log_prob.dim() > 0:
            log_prob_sum = log_prob.sum()
        else:
            log_prob_sum = log_prob

        # Compute gradient of log probability with respect to x
        score = torch.autograd.grad(
            outputs=log_prob_sum,
            inputs=x_var,
            create_graph=create_graph,
            retain_graph=create_graph,  # TODO: make this an option
        )[0]
        return score

    def sample(self, n_samples: int, t: float = 0.0) -> torch.Tensor:
        """
        Sample from the GMM at time t.

        Args:
            n_samples: number of samples
            t: time (default 0 = data distribution)

        Returns:
            samples: [n_samples, 1]
        """
        device = self.mu1.device
        t_tensor = torch.tensor(t, device=device)

        # Sample component assignments
        z = torch.rand(n_samples, device=device) < self.weight1

        # Get marginal parameters
        if hasattr(self.scheduler, "marginal_prob"):
            # CosineVPSDE or other SDE with marginal_prob method
            ones = torch.ones(1, 1, device=device)
            t_batch = t_tensor.unsqueeze(0)
            alpha_t, sigma_t = self.scheduler.marginal_prob(x=ones, t=t_batch)
            alpha_t = alpha_t.squeeze()
            sigma_t = sigma_t.squeeze()
        else:
            # BetaSchedule fallback
            alpha_t, sigma_t = self.scheduler.get_alpha_t_sigma_t(t_tensor)

        mu1_t = alpha_t * self.mu1
        mu2_t = alpha_t * self.mu2
        sigma1_t = torch.sqrt(self.sigma1**2 * alpha_t**2 + sigma_t**2)
        sigma2_t = torch.sqrt(self.sigma2**2 * alpha_t**2 + sigma_t**2)

        # Sample from each component
        samples1 = mu1_t + sigma1_t * torch.randn(n_samples, 1, device=device)
        samples2 = mu2_t + sigma2_t * torch.randn(n_samples, 1, device=device)

        # Mix samples based on component assignments
        samples = torch.where(z.unsqueeze(-1), samples1, samples2)
        return samples

    def energy(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return negative log probability."""
        return -self.log_prob(x, t)


class ToyPosCV(CollectiveVariable):
    """Minimal CV that extracts the first coordinate (scaled) from a ChemGraph.

    The bioemu steering stack expects CVs to operate on `ChemGraph` objects; for the
    1D toy we simply read the first scalar position. The positional inputs arrive as
    Angstrom in the steering code (they are multiplied by 10 prior to the call),
    so we divide by 10 to recover the original nm-scale variable used in sampling.
    """

    def compute_batch(self, pos: torch.Tensor, sequence: str) -> torch.Tensor:
        # pos shape: [batch_size, L, D]; toy uses L=1, D=1
        vals = pos.reshape(pos.shape[0], -1)[:, 0]  # scale back to nm
        return vals