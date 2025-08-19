import matplotlib.pyplot as plt
import torch
import numpy as np

plt.style.use('default')


def potential_loss_fn(x, target, tolerance, slope, max_value, order):
    """
    Flat-bottom loss for continuous variables using torch.abs and torch.relu.

    Args:
            x (Tensor): Input tensor.
            target (float or Tensor): Target value.
            tolerance (float): Flat region width around target.
            slope (float): Slope outside tolerance.
            max_value (float): Maximum loss value outside tolerance.

    Returns:
            Tensor: Loss values.
    """
    diff = torch.abs(x - target)
    # Only penalize values outside the tolerance
    penalty = (slope * torch.relu(diff - tolerance))**order
    # Cap the penalty at max_value
    loss = torch.clamp(penalty, max=max_value)
    return loss


x_vals = torch.linspace(0, 7, 500)
target = 3.8

tolerances = [0.1, 0.3, 0.5]
slopes = [1.0, 2.0, 5.0]
max_values = [2.0, 4.0]

fig, axs = plt.subplots(len(tolerances), len(slopes), figsize=(12, 8), sharex=True, sharey=True)
for i, tol in enumerate(tolerances):
    for j, slope in enumerate(slopes):
        for order in [1, 2]:
            for max_val in max_values:
                loss = potential_loss_fn(x_vals, target, tol, slope, max_val, order)
                axs[i, j].plot(x_vals.numpy(), loss.numpy(), label=f"max={max_val}")
            axs[i, j].set_title(f"tol={tol}, slope={slope}")
            axs[i, j].axvline(target, color='gray', linestyle='--', alpha=0.5)
            axs[i, j].legend()
            axs[i, j].set_xlabel("x")
            axs[i, j].set_ylabel("loss")

plt.tight_layout()
plt.show()
