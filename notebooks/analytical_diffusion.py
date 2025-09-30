"""
Analytical Diffusion with Time-Dependent Gaussian Mixture Model

This module implements a time-dependent Gaussian Mixture Model (GMM) that:
1. Starts with two differently weighted modes at t=0
2. Evolves through the forward SDE: dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
3. Ends in a unimodal normal distribution at t=1

The score is computed using autograd of the log probability.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from torch.distributions import Normal, MixtureSameFamily, Categorical

from tqdm import tqdm

# Import potential_loss_fn from bioemu steering
import sys
sys.path.append('/home/luwinkler/bioemu/src')
from bioemu.steering import potential_loss_fn

# Set matplotlib to use white background
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BetaSchedule:
    """
    Beta schedule for the forward SDE:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
    
    With β(t) = β_min + t(β_max - β_min)
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β(t) = β_min + t(β_max - β_min)"""
        t = t.to(device)
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integrated_beta(self, t: torch.Tensor) -> torch.Tensor:
        """∫₀ᵗ β(s) ds = β_min * t + (β_max - β_min) * t²/2"""
        t = t.to(device)
        return self.beta_min * t + (self.beta_max - self.beta_min) * t ** 2 / 2
    
    def get_alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """Signal coefficient: exp(-1/2 * ∫₀ᵗ β(s) ds)"""
        int_beta = self.integrated_beta(t)
        return torch.exp(-0.5 * int_beta)
    
    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient: √(1 - exp(-∫₀ᵗ β(s) ds))"""
        int_beta = self.integrated_beta(t)
        return torch.sqrt(1 - torch.exp(-int_beta))


class TimeDependentGMM:
    """Time-dependent Gaussian Mixture Model for analytical diffusion"""
    
    def __init__(
        self,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
        weight1: float = 0.7,
        beta_schedule: Optional[BetaSchedule] = None
    ):
        """
        Initialize time-dependent GMM
        
        Args:
            mu1: Mean of first mode [d]
            mu2: Mean of second mode [d] 
            sigma1: Standard deviation of first mode
            sigma2: Standard deviation of second mode
            weight1: Weight of first mode (weight2 = 1 - weight1)
            beta_schedule: Beta schedule for the forward SDE
        """
        self.mu1 = mu1.to(device)
        self.mu2 = mu2.to(device)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.weight1 = weight1
        self.weight2 = 1.0 - weight1
        self.dim = mu1.shape[-1]
        
        if beta_schedule is None:
            beta_schedule = BetaSchedule()
        self.schedule = beta_schedule
    
    def q0_sample(self, n_samples: int) -> torch.Tensor:
        """Sample from initial distribution q(x_0)"""
        # Sample mixture components
        component_samples = torch.rand(n_samples, device=device) < self.weight1
        
        # Sample from each component
        samples1 = torch.randn(n_samples, self.dim, device=device) * self.sigma1 + self.mu1.to(device)
        samples2 = torch.randn(n_samples, self.dim, device=device) * self.sigma2 + self.mu2.to(device)
        
        # Combine based on component assignment
        samples = torch.where(component_samples.unsqueeze(-1), samples1, samples2)
        return samples
    
    def q0_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability of initial distribution q(x_0)"""
        x = x.to(device)
        # Log probabilities for each component
        log_prob1 = Normal(self.mu1.to(device), self.sigma1).log_prob(x).sum(dim=-1)
        log_prob2 = Normal(self.mu2.to(device), self.sigma2).log_prob(x).sum(dim=-1)
        
        # Log-sum-exp for mixture
        log_weights = torch.log(torch.tensor([self.weight1, self.weight2], device=device))
        log_probs = torch.stack([log_prob1 + log_weights[0], log_prob2 + log_weights[1]], dim=-1)
        return torch.logsumexp(log_probs, dim=-1)
    
    def qt_mean_and_var(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of q(x_t | x_0)"""
        x0 = x0.to(device)
        t = t.to(device)
        alpha_t = self.schedule.get_alpha_t(t).to(device)
        sigma_t = self.schedule.get_sigma_t(t).to(device)
        
        # Expand dimensions for broadcasting
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.unsqueeze(0)
        if sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
            
        while alpha_t.dim() < x0.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.dim() < x0.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        
        mean = alpha_t * x0
        var = sigma_t ** 2
        return mean, var
    
    def qt_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t | x_0)"""
        mean, var = self.qt_mean_and_var(x0, t)
        noise = torch.randn_like(x0, device=device)
        return mean + torch.sqrt(var) * noise
    
    def qt_log_prob(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Log probability of q(x_t) - marginal distribution at time t
        
        This is computed by marginalizing over x0:
        q(x_t) = ∫ q(x_t | x_0) q(x_0) dx_0
        
        For our GMM, this becomes:
        q(x_t) = Σ_k w_k * N(x_t; α_t * μ_k, σ_t^2 + α_t^2 * σ_k^2)
        """
        xt = xt.to(device)
        t = t.to(device)
        alpha_t = self.schedule.get_alpha_t(t).to(device)
        sigma_t = self.schedule.get_sigma_t(t).to(device)
        
        # Expand dimensions
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.unsqueeze(0)
        if sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
            
        while alpha_t.dim() < xt.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        while sigma_t.dim() < xt.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        
        # Means and variances for each component at time t
        mean1_t = alpha_t * self.mu1.to(device)
        mean2_t = alpha_t * self.mu2.to(device)
        var1_t = sigma_t ** 2 + (alpha_t * self.sigma1) ** 2
        var2_t = sigma_t ** 2 + (alpha_t * self.sigma2) ** 2
        
        # Log probabilities for each component
        log_prob1 = Normal(mean1_t, torch.sqrt(var1_t)).log_prob(xt).sum(dim=-1)
        log_prob2 = Normal(mean2_t, torch.sqrt(var2_t)).log_prob(xt).sum(dim=-1)
        
        # Log-sum-exp for mixture
        log_weights = torch.log(torch.tensor([self.weight1, self.weight2], device=device))
        log_probs = torch.stack([log_prob1 + log_weights[0], log_prob2 + log_weights[1]], dim=-1)
        return torch.logsumexp(log_probs, dim=-1)
    
    def compute_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute score function using autograd: ∇_x log q(x_t)
        
        The score is computed by taking the gradient of the log probability
        with respect to the input x using automatic differentiation.
        """
        # Ensure x requires gradients and is on device
        x_copy = x.clone().detach().to(device).requires_grad_(True)
        t = t.to(device)
        
        # Compute log probability
        log_prob = self.qt_log_prob(x_copy, t)
        
        # Handle batch dimension - sum log probabilities to get scalar for autograd
        if log_prob.dim() > 0:
            log_prob_sum = log_prob.sum()
        else:
            log_prob_sum = log_prob
        
        # Compute gradient using autograd
        score = torch.autograd.grad(
            outputs=log_prob_sum,
            inputs=x_copy,
            create_graph=False,
            retain_graph=False
        )[0]
        
        return score
    
    def verify_score_numerically(self, x: torch.Tensor, t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Verify autograd score using numerical differentiation"""
        x = x.to(device)
        t = t.to(device)
        scores = []
        
        for i in range(x.shape[-1]):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[..., i] += eps
            x_minus[..., i] -= eps
            
            log_prob_plus = self.qt_log_prob(x_plus, t)
            log_prob_minus = self.qt_log_prob(x_minus, t)
            
            score_i = (log_prob_plus - log_prob_minus) / (2 * eps)
            scores.append(score_i)
        
        return torch.stack(scores, dim=-1)


def visualize_evolution(gmm: TimeDependentGMM, n_samples: int = 1000, n_timesteps: int = 6):
    """Visualize the evolution of the GMM through time"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Sample initial points
    x0_samples = gmm.q0_sample(n_samples)
    
    timesteps = torch.linspace(0, 1, n_timesteps)
    
    for i, t in enumerate(timesteps):
        # Forward diffusion samples
        xt_samples = gmm.qt_sample(x0_samples, t)
        
        axes[i].scatter(xt_samples[:, 0], xt_samples[:, 1], alpha=0.6, s=1)
        axes[i].set_title(f't = {t:.2f}')
        axes[i].set_xlim(-8, 8)
        axes[i].set_ylim(-8, 8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/luwinkler/bioemu/notebooks/outputs/gmm_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_score_computation():
    """Test the autograd score computation"""
    # Create 2D GMM with beta schedule
    mu1 = torch.tensor([-2.0, 1.0])
    mu2 = torch.tensor([3.0, -1.5])
    beta_schedule = BetaSchedule(beta_min=0.1, beta_max=20.0)
    gmm = TimeDependentGMM(mu1, mu2, sigma1=0.8, sigma2=1.2, weight1=0.3, beta_schedule=beta_schedule)
    
    # Test points and time
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    t = torch.tensor([0.3, 0.5, 0.8])
    
    # Compute autograd score
    autograd_score = gmm.compute_score(x, t)
    
    # Verify with numerical differentiation
    numerical_score = gmm.verify_score_numerically(x, t)
    
    print("Autograd score:")
    print(autograd_score)
    print("\nNumerical score:")
    print(numerical_score)
    print("\nDifference:")
    print(torch.abs(autograd_score - numerical_score))
    print(f"\nMax absolute difference: {torch.max(torch.abs(autograd_score - numerical_score)):.6f}")
    
    return autograd_score, numerical_score





def plot_score_function(gmm: TimeDependentGMM, t: float = 0.3):
    """Plot the score function for a given time"""
    x_range = torch.linspace(-6, 6, 100).unsqueeze(-1)
    x_range.requires_grad_(True)
    
    t_tensor = torch.full((100,), t)
    
    # Compute score using autograd
    scores = []
    for i in range(len(x_range)):
        x_single = x_range[i:i+1]
        t_single = t_tensor[i:i+1]
        score = gmm.compute_score(x_single, t_single)
        scores.append(score.item())
    
    scores = torch.tensor(scores)
    
    # Also compute log probability for reference
    log_probs = gmm.qt_log_prob(x_range, t_tensor)
    probs = torch.exp(log_probs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot probability density
    ax1.plot(x_range.detach().squeeze().detach().numpy(), probs.detach().numpy(), 'b-', linewidth=2)
    ax1.set_title(f'Probability Density at t={t}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x_t)')
    ax1.grid(True, alpha=0.3)
    
    # Plot score function
    ax2.plot(x_range.detach().squeeze().numpy(), scores.numpy(), 'r-', linewidth=2)
    ax2.set_title(f'Score Function ∇_x log p(x_t) at t={t}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Score')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/home/luwinkler/bioemu/notebooks/outputs/score_function_t{t}.png', dpi=150, bbox_inches='tight')
    plt.show()


def forward_diffusion(gmm: TimeDependentGMM, x: torch.Tensor = None, t: float = None, 
                               num_samples: int = None, n_steps: int = 100):
    """
    Simulate the forward diffusion process using Euler-Maruyama:
    dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
    
    Args:
        gmm: Time-dependent GMM model
        x: Initial samples [n_samples, 1] at time t (optional if num_samples provided)
        t: Starting time (0 <= t <= 1) (optional if num_samples provided)
        num_samples: Number of samples to generate from initial GMM at t=0 (optional if x,t provided)
        n_steps: Number of integration steps
        
    Returns:
        trajectory: [n_steps+1, n_samples, 1] - trajectory to terminal distribution
    """
    # XOR check: exactly one of (x and t) or num_samples should be provided
    has_x_and_t = (x is not None and t is not None)
    has_num_samples = (num_samples is not None)
    
    if not (has_x_and_t ^ has_num_samples):
        raise ValueError("Must provide either num_samples OR both (x, t), but not both or neither.")
    if num_samples is not None:
        # Mode 1: Sample from initial GMM distribution at t=0
        if x is not None or t is not None:
            raise ValueError("Cannot specify both num_samples and (x, t). Use either num_samples OR (x, t).")
        x = gmm.q0_sample(num_samples)
        t = 0.0
    elif x is not None and t is not None:
        # Mode 2: Start from provided samples at given time
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Ensure [n_samples, 1] shape
    else:
        raise ValueError("Must provide either num_samples OR both (x, t).")
    
    if t >= 1.0:
        # Already at terminal time, return input
        return x.unsqueeze(0)
    
    x = x.clone().to(device)
    n_samples = x.shape[0]
    
    # Time span from t to 1
    time_span = 1.0 - t
    dt = time_span / n_steps
    trajectory = [x.clone()]
    
    for step in tqdm(range(n_steps)):
        current_t = t + step * dt
        t_tensor = torch.full((n_samples,), current_t, device=device)
        
        # Get beta value
        beta_t = gmm.schedule.beta(t_tensor).to(device).unsqueeze(-1)  # [n_samples, 1]
        
        # Forward SDE: dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)
        noise = torch.randn_like(x, device=device)
        
        # Euler-Maruyama step
        x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=device)) * noise
        trajectory.append(x.clone())
    
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    
    # Create time indices for each step
    time_indices = torch.linspace(t, 1.0, n_steps + 1, device=device)  # [n_steps+1]
    
    return trajectory_tensor, time_indices


def reverse_diffusion(gmm: TimeDependentGMM, x: torch.Tensor = None, t: float = None,
                               num_samples: int = None, n_steps: int = 100, 
                               potentials: list = None, steering_config: dict = None):
    """
    Simulate the reverse diffusion process using the score function:
    dX_t = [1/2 * β(t) * X_t + β(t) * ∇_x log p_t(x)] dt + √β(t) dW_t
    
    Args:
        gmm: Time-dependent GMM model
        x: Initial samples [n_samples, 1] at time t (optional if num_samples provided)
        t: Starting time (0 <= t <= 1) (optional if num_samples provided)
        num_samples: Number of samples to generate from N(0,1) at t=1 (optional if x,t provided)
        n_steps: Number of integration steps
        potentials: List of potential functions for steering
        steering_config: Dict with keys: num_particles, start, resampling_freq
        
    Returns:
        trajectory: [n_steps+1, n_samples, 1] - trajectory to terminal distribution
    """
    if num_samples is not None:
        # Mode 1: Sample from terminal Normal distribution at t=1
        if x is not None or t is not None:
            raise ValueError("Cannot specify both num_samples and (x, t). Use either num_samples OR (x, t).")
        x = torch.randn(num_samples, 1).to(device)
        # If steering is enabled, tile each sample to create num_particles copies
		
        if steering_config:
            num_particles = steering_config['num_particles']
            x = x.repeat_interleave(num_particles, dim=0)  # [num_samples * num_particles, 1]
        t = 1.0
    elif x is not None and t is not None:
        # Mode 2: Start from provided samples at given time
        x = x.to(device)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # Ensure [n_samples, 1] shape
    else:
        raise ValueError("Must provide either num_samples OR both (x, t).")
    
    if t <= 0.0:
        # Already at terminal time, return input
        return x.unsqueeze(0)
    
    x = x.clone().to(device)
    n_samples = x.shape[0]
    
    # Time span from t to 0 (backwards)
    time_span = t
    dt = time_span / n_steps
    trajectory = [x.clone()]
    previous_energy = None
    
    for step in tqdm(range(n_steps)):
        current_t = t - step * dt  # Go backwards in time
        t_tensor = torch.full((n_samples,), current_t, device=device)
        
        # Get beta value
        beta_t = gmm.schedule.beta(t_tensor).to(device).unsqueeze(-1)  # [n_samples, 1]
        
        # Compute score for entire batch at once
        score_batch = gmm.compute_score(x, t_tensor)  # [n_samples, 1]
        
        # Reverse SDE: dX_t = [1/2 * β(t) * X_t + β(t) * score] dt + √β(t) dW_t
        drift = 0.5 * beta_t * x + beta_t * score_batch
        diffusion = torch.sqrt(beta_t)
        noise = torch.randn_like(x, device=device)
        
        # Euler-Maruyama step (backwards in time)
        x = x + drift * dt + diffusion * torch.sqrt(torch.tensor(dt, device=device)) * noise
        trajectory.append(x.clone())
        
        # Steering functionality (same logic as denoiser)
        if steering_config and potentials:
            # Extract clean data x0 from score using Tweedie's formula
            # For VP-SDE: x0 = (x + beta_t * score) / sqrt(alpha_t)
            alpha_t = gmm.schedule.get_alpha_t(t_tensor).to(device).unsqueeze(-1)  # [n_samples, 1]
            sigma_t = gmm.schedule.get_sigma_t(t_tensor).to(device).unsqueeze(-1)  # [n_samples, 1]
            x0_pred = (x + sigma_t * score_batch) / torch.sqrt(alpha_t)
            # plt.hist(x0_pred.squeeze(-1).cpu().numpy(), bins=100, density=True)
            # plt.title(f'Predicted clean data at t={current_t}')
            # plt.show()
            
            # Evaluate potentials on predicted clean data
            energies = [pot(x0_pred.squeeze(-1)) for pot in potentials]
            total_energy = torch.stack(energies, dim=-1)  # [n_samples]
            
            if steering_config['num_particles'] > 1:
                start_step = int(n_steps * steering_config['start'])
                resample_freq = steering_config['resampling_freq']
                
                if start_step <= step < (n_steps - 2) and step % resample_freq == 0:
                    x, total_energy = resample_particles(x, total_energy, previous_energy, steering_config)
                
                    previous_energy = total_energy if steering_config['previous_energy'] else None
    
    trajectory_tensor = torch.stack(trajectory)  # [n_steps+1, n_samples, 1]
    
    # Create time indices for each step (going backwards from t to 0)
    time_indices = torch.linspace(t, 0.0, n_steps + 1, device=device)  # [n_steps+1]
    
    return trajectory_tensor, time_indices


def resample_particles(x, energy, previous_energy=None, steering_config=None, final=False):
    """
    Resampling for particle filter - each original sample has num_particles copies.
    Resampling is done within each group of particles for each sample.
    """
    num_particles = steering_config['num_particles']
    total_size = x.shape[0]
    num_samples = total_size // num_particles
    
    # Reshape to [num_samples, num_particles, ...]
    x_grouped = x.view(num_samples, num_particles, -1)
    energy_grouped = energy.view(num_samples, num_particles)
    
    if previous_energy is not None:
        prev_energy_grouped = previous_energy.view(num_samples, num_particles)
        resample_logprob = prev_energy_grouped - energy_grouped
    else:
        resample_logprob = -energy_grouped
    
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
    # exp(log(softmax(x))) = exp(log(exp(x_i)/sum(exp(x_i))))
    # log_softmax uses numerical stabilization by shifting the values to avoid overflow (absolute value of energy is meaningless)
    probs = torch.exp(torch.nn.functional.log_softmax(resample_logprob, dim=-1))  # [num_samples, num_particles]
    # probs = torch.exp(resample_logprob)
    
    # Generate multinomial indices for all samples
    indices = torch.multinomial(probs, num_samples=num_particles, replacement=True)  # [num_samples, num_particles]
    
    # Use advanced indexing to resample all particles at once
    resampled_x = torch.stack([x_grouped[i, indices[i]] for i in range(num_samples)]).view(total_size, -1)
    resampled_energy = torch.stack([energy_grouped[i, indices[i]] for i in range(num_samples)]).view(total_size)
    
    
    return resampled_x, resampled_energy


def visualize_diffusion_trajectories(forward_data, reverse_data, n_display: int = 50):
    """Visualize individual particle trajectories with time on x-axis and position on y-axis"""
    # Handle both old format (just trajectory) and new format (trajectory, times)
    if isinstance(forward_data, tuple):
        forward_traj, forward_times = forward_data
        reverse_traj, reverse_times = reverse_data
    else:
        # Backward compatibility
        forward_traj = forward_data
        reverse_traj = reverse_data
        n_steps = forward_traj.shape[0]
        forward_times = torch.linspace(0, 1, n_steps)
        reverse_times = torch.linspace(1, 0, n_steps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot some individual trajectories
    for i in range(min(n_display, forward_traj.shape[1])):
        # Forward trajectories: time on x-axis, position on y-axis
        ax1.plot(forward_times.cpu().numpy(), forward_traj[:, i, 0].cpu().numpy(), 'b-', alpha=0.4, linewidth=1.0)
        
        # Reverse trajectories: time on x-axis, position on y-axis
        ax2.plot(reverse_times.cpu().numpy(), reverse_traj[:, i, 0].cpu().numpy(), 'r-', alpha=0.4, linewidth=1.0)

    
    # Forward plot formatting
    ax1.set_title('Forward Diffusion Trajectories\n(GMM → Normal)', fontsize=14)
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel('Position x', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Reverse plot formatting
    ax2.set_title('Reverse Diffusion Trajectories\n(Normal → GMM)', fontsize=14)
    ax2.set_xlabel('Time t (0=start, 1=end of reverse process)', fontsize=12)
    ax2.set_ylabel('Position x', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    # Add annotations
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.text(0.05, 4, 'Start: Bimodal GMM', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.text(0.75, 0.5, 'End: N(0,1)', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.5, 'Start: N(0,1)', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    ax2.text(0.6, 3, 'End: Bimodal GMM', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.savefig('/home/luwinkler/bioemu/notebooks/outputs/diffusion_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_trajectory_heatmap(forward_traj, reverse_traj, forward_times=None, reverse_times=None):
    """Plot trajectory density as heatmaps over time"""
    # Handle different calling patterns
    if forward_times is None or reverse_times is None:
        # Backward compatibility - generate default time arrays
        n_steps = forward_traj.shape[0]
        forward_times = torch.linspace(0, 1, n_steps)
        reverse_times = torch.linspace(1, 0, n_steps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    n_steps = forward_traj.shape[0]
    
    # Create position bins
    x_min, x_max = -6, 6
    x_bins = torch.linspace(x_min, x_max, 100)
    
    # Forward trajectory heatmap
    forward_density = torch.zeros(n_steps, len(x_bins)-1)
    for t_idx in range(n_steps):
        positions = forward_traj[t_idx, :, 0].cpu()  # Move to CPU for histogram
        hist, _ = torch.histogram(positions, bins=x_bins.cpu(), density=True)
        forward_density[t_idx] = hist
    
    # Reverse trajectory heatmap
    reverse_density = torch.zeros(n_steps, len(x_bins)-1)
    for t_idx in range(n_steps):
        positions = reverse_traj[t_idx, :, 0].cpu()  # Move to CPU for histogram
        hist, _ = torch.histogram(positions, bins=x_bins.cpu(), density=True)
        reverse_density[t_idx] = hist
    
    # Plot heatmaps
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    T_forward, X_forward = torch.meshgrid(forward_times.cpu(), x_centers, indexing='ij')
    T_reverse, X_reverse = torch.meshgrid(reverse_times.cpu(), x_centers, indexing='ij')
    
    im1 = ax1.contourf(T_forward.numpy(), X_forward.numpy(), forward_density.numpy(), levels=50, cmap='Blues')
    ax1.set_title('Forward Diffusion Density\n(Time vs Position)', fontsize=14)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Position x')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    im2 = ax2.contourf(T_reverse.numpy(), X_reverse.numpy(), reverse_density.numpy(), levels=50, cmap='Reds')
    ax2.set_title('Reverse Diffusion Density\n(Time vs Position)', fontsize=14)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Position x')
    plt.colorbar(im2, ax=ax2, label='Density')
    
    plt.tight_layout()
    plt.savefig('/home/luwinkler/bioemu/notebooks/outputs/trajectory_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_processes(gmm: TimeDependentGMM, forward_data=None, reverse_data=None, n_display=50, potentials=None):
    """
    Plot diffusion processes with consistent layout.
    
    Layout: Always 2x5 grid (forward in row 0, reverse in row 1)
    - Column 0: t=0 distributions (GMM)  
    - Column 2: Trajectory evolution
    - Column 4: t=1 distributions (Normal)
    
    Args:
        potentials: Optional list of potential functions to plot in terminal distributions
    """
    # Validate inputs
    has_forward = forward_data is not None
    has_reverse = reverse_data is not None
    
    if not has_forward and not has_reverse:
        raise ValueError("At least one of forward_data or reverse_data must be provided")
    
    # Unpack data
    if has_forward:
        if isinstance(forward_data, tuple):
            forward_traj, forward_times = forward_data
            plot_forward_trajectories = True
        else:
            forward_traj = forward_data
            plot_forward_trajectories = False
    
    if has_reverse:
        if isinstance(reverse_data, tuple):
            reverse_traj, reverse_times = reverse_data
            plot_reverse_trajectories = True
        else:
            reverse_traj = reverse_data
            plot_reverse_trajectories = False
    
    # Create consistent 2x5 layout
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.1, 2, 0.1, 1], height_ratios=[1, 1])
    
    # Extract terminal distributions
    if has_forward:
        forward_initial = forward_traj[0].squeeze()  # t=0: GMM
        forward_final = forward_traj[-1].squeeze()   # t=1: N(0,1)
    
    if has_reverse:
        reverse_initial = reverse_traj[0].squeeze()  # t=1: N(0,1) 
        reverse_final = reverse_traj[-1].squeeze()   # t=0: GMM
    
    # Create analytical curves
    x_range = torch.linspace(-6, 6, 200)
    x_range_unsqueezed = x_range.unsqueeze(-1)
    
    # Analytical distributions
    log_probs_gmm = gmm.q0_log_prob(x_range_unsqueezed)
    probs_gmm = torch.exp(log_probs_gmm)
    probs_normal = torch.exp(-0.5 * x_range**2) / torch.sqrt(2 * torch.pi * torch.ones(1))
    
    # Plot forward process (row 0) if available
    if has_forward:
        # t=0 distribution (left, col 0)
        ax_t0_forward = fig.add_subplot(gs[0, 0])
        hist_counts, hist_bins = np.histogram(forward_initial.cpu().numpy(), bins=30, density=True, range=(-6, 6))
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        ax_t0_forward.barh(bin_centers, hist_counts, height=hist_bins[1]-hist_bins[0], 
                          alpha=0.6, color='skyblue', label='Forward t=0')
        ax_t0_forward.plot(probs_gmm.cpu().numpy(), x_range.cpu().numpy(), 'b-', linewidth=3, label='Analytical GMM')
        ax_t0_forward.set_ylim(-6, 6)
        ax_t0_forward.set_xlabel('Density')
        ax_t0_forward.set_ylabel('Position x')
        ax_t0_forward.set_title('t=0: GMM\n(Start)', fontweight='bold', fontsize=11)
        ax_t0_forward.grid(True, alpha=0.3)
        ax_t0_forward.legend(fontsize=9)
        
        # Trajectory (center, col 2)
        ax_traj_forward = fig.add_subplot(gs[0, 2])
        if plot_forward_trajectories:
            for i in range(min(n_display, forward_traj.shape[1])):
                ax_traj_forward.plot(forward_times.cpu().numpy(), forward_traj[:, i, 0].cpu().numpy(), 
                                   'b-', alpha=0.3, linewidth=0.8)
            
        ax_traj_forward.set_title('Forward Process: GMM → N(0,1)', fontweight='bold', fontsize=12)
        ax_traj_forward.set_xlabel('Time t (0=start from data, 1=end at noise)')
        ax_traj_forward.set_ylabel('Position x')
        ax_traj_forward.set_xlim(0, 1)
        ax_traj_forward.set_ylim(-8, 8)
        ax_traj_forward.grid(True, alpha=0.3)
        if plot_forward_trajectories:
            ax_traj_forward.legend()
        
        # t=1 distribution (right, col 4)
        ax_t1_forward = fig.add_subplot(gs[0, 4])
        hist_counts, hist_bins = np.histogram(forward_final.cpu().numpy(), bins=30, density=True, range=(-6, 6))
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        ax_t1_forward.barh(bin_centers, hist_counts, height=hist_bins[1]-hist_bins[0], 
                          alpha=0.6, color='lightcoral', label='Forward t=1')
        ax_t1_forward.plot(probs_normal.cpu().numpy(), x_range.cpu().numpy(), 'r-', linewidth=3, label='Analytical N(0,1)')
        ax_t1_forward.set_ylim(-8, 8)
        ax_t1_forward.set_xlabel('Density')
        ax_t1_forward.set_ylabel('Position x')
        ax_t1_forward.set_title('t=1: N(0,1)\n(End)', fontweight='bold', fontsize=11)
        ax_t1_forward.grid(True, alpha=0.3)
        ax_t1_forward.legend(fontsize=9)
    
    # Plot reverse process (row 1) if available
    if has_reverse:
        # t=0 distribution (right side for reverse, col 4)
        ax_t0_reverse = fig.add_subplot(gs[1, 4])
        hist_counts, hist_bins = np.histogram(reverse_final.cpu().numpy(), bins=30, density=True, range=(-6, 6))
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        ax_t0_reverse.barh(bin_centers, hist_counts, height=hist_bins[1]-hist_bins[0], 
                          alpha=0.3, color='red', label='Reverse t=0')
        ax_t0_reverse.plot(probs_gmm.cpu().numpy(), x_range.cpu().numpy(), 'b-', linewidth=3, label='Analytical GMM')
        
        # Plot potentials if provided
        if potentials is not None:
            # Create a second x-axis for potential energy
            ax_potential = ax_t0_reverse.twiny()
            for i, potential in enumerate(potentials):
                # Evaluate potential over x_range
                x_input = x_range.unsqueeze(-1)  # Add dimension for potential function
                potential_values = potential(x_input).cpu().numpy()
                
                # Normalize potential for plotting (scale to fit with density)
                max_density = probs_gmm.max().cpu().numpy()
                potential_normalized = potential_values / potential_values.max() * max_density * 0.5
                
                # Plot potential as a line
                ax_potential.plot(potential_normalized, x_range.cpu().numpy(), 
                                linestyle='--', linewidth=2, alpha=0.8,
                                label=f'Potential {i+1}', color=f'C{i+2}')
                
                # Calculate and plot Boltzmann distribution from the potential
                kT = 1.0  # Temperature
                
                # Evaluate potential at bin centers directly
                bin_centers_tensor = torch.tensor(bin_centers, dtype=torch.float32).unsqueeze(-1)
                potential_at_bins = potential(bin_centers_tensor).cpu().numpy().flatten()
                
                # Calculate Boltzmann distribution at bin centers
                boltzmann_at_bins = np.exp(-potential_at_bins / kT)
                # Normalize to be a proper probability density
                dx_bins = hist_bins[1] - hist_bins[0]
                boltzmann_normalized = boltzmann_at_bins / (boltzmann_at_bins.sum() * dx_bins)
                
                # Plot Boltzmann distribution using histogram bins
                ax_t0_reverse.step(boltzmann_normalized, bin_centers, 
                                 where='mid', alpha=0.6, color='green', 
                                 label=f'Boltzmann {i+1}')
                
                # Evaluate analytical GMM at bin centers
                gmm_at_bins = torch.exp(gmm.q0_log_prob(bin_centers_tensor)).cpu().numpy().flatten()
                
                # Multiply GMM with Boltzmann distribution (analytical posterior)
                analytical_posterior = gmm_at_bins * boltzmann_at_bins
                # Renormalize to be a proper probability density
                analytical_posterior_normalized = analytical_posterior / (analytical_posterior.sum() * dx_bins)
                
                # Plot analytical posterior
                ax_t0_reverse.step(analytical_posterior_normalized, bin_centers, 
                                 where='mid', alpha=0.7, color='blue',
                                 label=f'Analytical Posterior {i+1}')

                
            
            ax_potential.set_xlabel('Normalized Potential Energy', color='red')
            ax_potential.tick_params(axis='x', labelcolor='red')
            ax_potential.legend(loc='upper left')
        
        ax_t0_reverse.set_ylim(-8, 8)
        ax_t0_reverse.set_xlim(0, 0.5)
        ax_t0_reverse.set_xlabel('Density')
        ax_t0_reverse.set_ylabel('Position x')
        ax_t0_reverse.set_title('t=0: GMM\n(End)', fontweight='bold', fontsize=11)
        ax_t0_reverse.grid(True, alpha=0.3)
        ax_t0_reverse.legend(fontsize=9)
        
        # Trajectory (center, col 2)
        ax_traj_reverse = fig.add_subplot(gs[1, 2])
        if plot_reverse_trajectories:
            # For reverse process, plot with original reverse_times (goes from t to 0)
            # This shows the actual reverse diffusion time progression
            for i in range(min(n_display, reverse_traj.shape[1])):
                ax_traj_reverse.plot(reverse_times.cpu().numpy(), reverse_traj[:, i, 0].cpu().numpy(), 
                                   'r-', alpha=0.3, linewidth=0.8)
            
        ax_traj_reverse.set_title('Reverse Process: N(0,1) → GMM', fontweight='bold', fontsize=12)
        ax_traj_reverse.set_xlabel('Time t (1=start from noise, 0=end at data)')
        ax_traj_reverse.set_ylabel('Position x')
        ax_traj_reverse.set_xlim(0, 1)
        ax_traj_reverse.set_ylim(-8, 8)
        ax_traj_reverse.grid(True, alpha=0.3)
        # Flip x-axis so time goes from 1 to 0 (left to right)
        ax_traj_reverse.invert_xaxis()
        if plot_reverse_trajectories:
            ax_traj_reverse.legend()
        
        # t=1 distribution (left side for reverse, col 0)
        ax_t1_reverse = fig.add_subplot(gs[1, 0])
        hist_counts, hist_bins = np.histogram(reverse_initial.cpu().numpy(), bins=30, density=True, range=(-6, 6))
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        ax_t1_reverse.barh(bin_centers, hist_counts, height=hist_bins[1]-hist_bins[0], 
                          alpha=0.6, color='orange', label='Reverse t=1')
        ax_t1_reverse.plot(probs_normal.cpu().numpy(), x_range.cpu().numpy(), 'r-', linewidth=3, label='Analytical N(0,1)')
        ax_t1_reverse.set_ylim(-8, 8)
        ax_t1_reverse.set_xlabel('Density')
        ax_t1_reverse.set_ylabel('Position x')
        ax_t1_reverse.set_title('t=1: N(0,1)\n(Start)', fontweight='bold', fontsize=11)
        ax_t1_reverse.grid(True, alpha=0.3)
        ax_t1_reverse.legend(fontsize=9)
    
    # Add title and annotations
    if has_forward and has_reverse:
        fig.suptitle('Diffusion Processes', ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.52, '→ Forward Process →', ha='center', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        fig.text(0.5, 0.02, '→ Reverse Process →', ha='center', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    elif has_forward:
        fig.suptitle('Forward Diffusion Process: GMM → N(0,1)', ha='center', fontsize=16, fontweight='bold')
    else:  # has_reverse
        fig.suptitle('Reverse Diffusion Process: N(0,1) → GMM', ha='center', fontsize=16, fontweight='bold')
    
    plt.subplots_adjust(top=0.92, bottom=0.08)
    plt.tight_layout()
    
    plt.show()
    
 



#%%
# Create output directory
import os
os.makedirs('/home/luwinkler/bioemu/notebooks/outputs', exist_ok=True)

# Create 1D GMM with two modes
mu1 = torch.tensor([-2.0], device=device)   # First mode
mu2 = torch.tensor([3.0], device=device)    # Second mode

# Create beta schedule: β(t) = β_min + t(β_max - β_min)
beta_schedule = BetaSchedule(beta_min=0.1, beta_max=20.0)

# Create GMM with different weights
gmm = TimeDependentGMM(
	mu1=mu1, 
	mu2=mu2,
	sigma1=0.8,
	sigma2=1.2, 
	weight1=0.7,  # 70% weight on first mode
	beta_schedule=beta_schedule
)

# print("=== 1D Time-Dependent Gaussian Mixture Model ===")
# print(f"Mode 1: μ={mu1.item():.1f}, σ={gmm.sigma1}, weight={gmm.weight1}")
# print(f"Mode 2: μ={mu2.item():.1f}, σ={gmm.sigma2}, weight={gmm.weight2}")
# print(f"Beta schedule: β(t) = {beta_schedule.beta_min} + t * {beta_schedule.beta_max - beta_schedule.beta_min}")
# print(f"Forward SDE: dX_t = -1/2 * β(t) * X_t dt + √β(t) dW_t")

# Test some beta values
print(f"\nBeta schedule values:")
for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
	t_tensor = torch.tensor([t_val], device=device)
	beta_t = beta_schedule.beta(t_tensor)
	alpha_t = beta_schedule.get_alpha_t(t_tensor)
	sigma_t = beta_schedule.get_sigma_t(t_tensor)
	print(f"  t={t_val:.2f}: β(t)={beta_t.item():.2f}, α_t={alpha_t.item():.4f}, σ_t={sigma_t.item():.4f}")

# Test score computation
# print("\n=== Testing Autograd Score (1D) ===")
# Test points and time for 1D
x = torch.tensor([[0.0], [1.0], [-1.0]], device=device)
t = torch.tensor([0.3, 0.5, 0.8], device=device)

# Compute autograd score
autograd_score = gmm.compute_score(x, t)
numerical_score = gmm.verify_score_numerically(x, t)


x_grid = torch.linspace(-6, 6, 100, device=device).unsqueeze(-1)

for t_val in [0.0, 0.3, 0.6, 1.0]:
	t_tensor = torch.tensor([t_val], device=device).expand(x_grid.shape[0])
	log_probs = gmm.qt_log_prob(x_grid, t_tensor)
	probs = torch.exp(log_probs)
	mean_val = (x_grid.squeeze() * probs).sum() / probs.sum()
	var_val = ((x_grid.squeeze() - mean_val)**2 * probs).sum() / probs.sum()
	# print(f"t={t_val:.1f}: mean={mean_val.item():.3f}, var={var_val.item():.3f}, max_prob={probs.max().item():.4f}")

# Test forward and reverse diffusion
# print("\n=== Forward and Reverse Diffusion Test ===")

n_samples = 512
n_steps = 200
print("Running forward diffusion...")
forward_trajectory, forward_times = forward_diffusion(gmm, num_samples=n_samples, n_steps=n_steps)
    
# Run reverse diffusion  
print("Running reverse diffusion...")
reverse_trajectory, reverse_times = reverse_diffusion(gmm, num_samples=n_samples, n_steps=n_steps)
print('done')

# Print time information for user reference
print(f"\nTime indices shape: {forward_times.shape}")
print(f"Forward time range: {forward_times[0].item():.3f} → {forward_times[-1].item():.3f}")
print(f"Reverse time range: {reverse_times[0].item():.3f} → {reverse_times[-1].item():.3f}")
print(f"Trajectory shape: {forward_trajectory.shape}")
print("Use forward_times and reverse_times arrays for plotting!")

#%%	

# Plot terminal distributions
print("\n=== Plotting Terminal Distributions (Rotated) ===")
plot_processes(gmm, (forward_trajectory, forward_times), (reverse_trajectory, reverse_times), n_display=50)

#%%

x= torch.randn(1000, 1) * 0.1 + 3
t= 0.5
reverse_trajectory, reverse_times = reverse_diffusion(gmm, x, t, n_steps=50)

plot_processes(gmm, forward_data=None, reverse_data=(reverse_trajectory, reverse_times), n_display=50)

# Example: Steering with harmonic potential
print("\n=== Reverse Diffusion with Steering ===")

steered_traj, steered_times = reverse_diffusion(
    gmm=gmm, 
    num_samples=2000,
    n_steps=50,
    # potentials=[harmonic_pot],
    # steering_config=steering_config
)

plot_processes(gmm, forward_data=(forward_trajectory, forward_times), reverse_data=(steered_traj, steered_times), n_display=500)


#%%

def create_harmonic_potential(
    target: float = -2.0, 
    tolerance: float = 0.1, 
    slope: float = 1.0, 
    max_value: float = 10.0, 
    order: float = 2.0, 
    linear_from: float = 2.0, 
    weight: float = 1.0
):

    def harmonic_potential(x: torch.Tensor) -> torch.Tensor:
        
        # Calculate potential using the flat-bottom loss function
        energy = potential_loss_fn(
            x=x,
            target=target,
            flat_bottom=tolerance,
            slope=slope,
            max_value=max_value,
            order=order,
            linear_from=linear_from
        )
        
        return weight * energy
    
    return harmonic_potential	

for target in torch.linspace(-6, 6, 5):
# Create a harmonic potential targeting the first GMM mode at x = -2.0
    harmonic_pot = create_harmonic_potential(
        target=target,      # Target the first GMM mode
        tolerance=0.2,    # Small flat region around target
        slope=0.3,        # Moderate slope
        max_value=1_000_000.0,    # Reasonable maximum energy
        order=2.0,        # Quadratic potential
        linear_from=1_000_000.0,  # Switch to linear after distance 1.0 from tolerance
        weight=1.0        # Unit weight
    )
    # Define steering configuration (same keywords as denoiser)
    steering_config = {

        'num_particles': 20,  # Multiple particles for resampling
        'start': 0.2,         # Start steering after 30% of steps
        'resampling_freq': 5,  # Resample every 5 steps
        'previous_energy': True
    }

    # Create potential targeting Mode 1
    # mode1_potential = create_harmonic_potential(target=-2.0, tolerance=0.2, slope=1.0, weight=1.0)

    steered_traj, steered_times = reverse_diffusion(
        gmm=gmm, 
        num_samples=2000,
        n_steps=50,
        potentials=[harmonic_pot],
        steering_config=steering_config
    )

    plot_processes(gmm, forward_data=(forward_trajectory, forward_times), reverse_data=(steered_traj, steered_times), potentials=[harmonic_pot], n_display=500)