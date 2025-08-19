import numpy as np
import torch
import matplotlib.pyplot as plt
from bioemu.steering import TerminiDistancePotential, potential_loss_fn

plt.style.use('default')

# Load the .npz file
# npz_path1 = "/home/luwinkler/bioemu/outputs/test_steering/FK_GYDPETGTWG_len:10/batch_0000000_0001024.npz"
steered = "/home/luwinkler/bioemu/outputs/test_steering/FK_GYDPETGTWG_len:10_steered/batch_0000000_0001024.npz"
non_steered = "/home/luwinkler/bioemu/outputs/test_steering/FK_GYDPETGTWG_len:10/batch_0000000_0001024.npz"

steered_data = np.load(steered)
non_steered = np.load(non_steered)

steered_pos, steered_rot = steered_data['pos'], steered_data['node_orientations']
steered_termini_distance = np.linalg.norm(steered_pos[:, 0] - steered_pos[:, -1], axis=-1)
steered_termini_distance = steered_termini_distance[steered_termini_distance < 5]  # Filter distances less than 10

non_steered_pos, non_steered_rot = non_steered['pos'], non_steered['node_orientations']
non_steered_termini_distance = np.linalg.norm(non_steered_pos[:, 0] - non_steered_pos[:, -1], axis=-1)
non_steered_termini_distance = non_steered_termini_distance[non_steered_termini_distance < 5]  # Filter distances less than 10

# Harmonic energy potential: E(x) = 0.5 * k * (x - x0)^2
target = 1.5
tolerance = 0.25
slope = 2
max_value = 10
order = 1
linear_from = 1.

bins = 50
x = np.linspace(0, 4, bins)
energy = lambda x: potential_loss_fn(torch.from_numpy(x), target=target, tolerance=tolerance, slope=slope, max_value=max_value, order=order, linear_from=linear_from).numpy()
pot = TerminiDistancePotential(
    target=target,
    tolerance=tolerance,
    slope=slope,
    max_value=max_value,
    order=order,
    weight=1.0
)
pot_energy = lambda pos: pot(N_pos=None, Ca_pos=pos, C_pos=None, O_pos=None, t=None, N=None)

custom_pos = torch.stack([torch.linspace(0, 5, 100), torch.zeros(100), torch.zeros(100)], dim=-1).unsqueeze(1)  # shape: [100, 3]
custom_pos = torch.concat([torch.zeros_like(custom_pos), custom_pos], dim=1)  # shape: [100, 2, 3]

pot_energy_x_location = np.linalg.norm(custom_pos[:, 0] - custom_pos[:, -1], axis=-1)
pot_energy = pot_energy(custom_pos).numpy().squeeze()  # Convert to numpy and squeeze

# Boltzmann distribution: p(x) ∝ exp(-E(x)/kT)
kT = 1.0
boltzmann = np.exp(-energy(x) / kT)
boltzmann /= boltzmann.sum() * (x[1] - x[0])  # Normalize

# Calculate center of mass for each sample in the batch
# center_of_mass = pos.mean(axis=1, keepdims=True)  # shape: [BS, 1, 3]
# squared_distances = np.sum((pos - center_of_mass) ** 2, axis=-1)  # shape: [BS, L]
# moment_of_gyration = np.sqrt(np.mean(squared_distances, axis=1))  # shape: [BS]
# moment_of_gyration = moment_of_gyration[moment_of_gyration < 5]  # Filter distances less than 10

# Compute empirical joint distribution between termini_distance and Boltzmann potential
# Bin the termini_distance using the same grid x
# termini_distance = np.random.uniform(0, 5, size=100000)  # Generate random distances for demonstration
steered_hist, _ = np.histogram(steered_termini_distance, bins=np.linspace(0, 5, bins + 1), density=True)
non_steered_hist, _ = np.histogram(non_steered_termini_distance, bins=np.linspace(0, 5, bins + 1), density=True)
# hist = np.ones_like(hist)
# hist /= hist.sum() * (x[1] - x[0])  # Normalize

# The joint distribution is the product of the empirical histogram and the Boltzmann distribution (both on x)
steered = non_steered_hist * boltzmann
steered /= steered.sum() * (x[1] - x[0])  # Normalize

plt.figure()
plt.plot(x, steered, label="Analytical Posterior", color='red')
plt.plot(x, energy(x), label="Potential", color='green')
plt.plot(x, boltzmann, label="Potential Distribution", color='green', ls='--')
# plt.plot(pot_energy_x_location, pot_energy, label="p(-E(x')) from TerminiDistancePotential")

# print("Moment of gyration (per sample):", moment_of_gyration)
plt.hist(steered_termini_distance, bins=x, label='Sampled Posterior', alpha=0.5, density=True, color='red')
plt.hist(non_steered_termini_distance, bins=x, label='Prior', alpha=0.5, density=True, color='blue')
# plt.hist(moment_of_gyration, bins=x, label='Moment of Gyration', alpha=0.5, density=True)
plt.ylim(0, 6)
plt.legend()
plt.tight_layout()
