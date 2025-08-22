# Working with Blob Storage in Feynman/EMU
# =======================================
#
# This script demonstrates how to download and work with data from blob storage
# using the blob_storage module from the feynman project.
#
# Key learnings:
#
# 1. The blob_storage module provides several functions to work with blob storage:
#    - uri_to_local(blob_uri): Converts a blob URI to a local path (doesn't download)
#    - download_file(blob_uri): Downloads a single file from blob storage
#    - download_dir(blob_uri): Downloads a directory from blob storage
#    - uri_file_exists(blob_uri): Checks if a file exists in blob storage
#
# 2. The CuratedMDDir class expects a specific directory structure:
#    - topology.pdb: The topology file
#    - dataset.json: Metadata about the dataset
#    - trajs/: Directory containing trajectory files
#
# 3. The load_simulation_from_files function can load trajectory fragments
#    which can then be joined using mdtraj.join()
#
# 4. The blob_storage module caches downloads in a local directory, so
#    subsequent requests for the same file will use the local copy

# Run it with the emu environment

# %%
from pdb import post_mortem
import matplotlib.pyplot as plt
import numpy as np

import mdtraj
import sys
import os
from pathlib import Path

# Add the feynman directory to the Python path
sys.path.append('/home/luwinkler')

from feynman.projects.emu.core.dataset.processing import load_simulation_from_files
from feynman.projects.emu.core.data.curated_md_dir import CuratedMDDir
from feynman.projects.emu.core.data import blob_storage

# Import from feynman project

# Define the blob URI for the directory containing MD data
blob_uri = "blob-storage://sampling0storage/curated-md/ONE_cath1/cath1_1bl0A02/"

# Get the local path for the blob URI (doesn't download anything)
local_path = blob_storage.uri_to_local(blob_uri)

# Download the directory if it doesn't exist locally
if not os.path.exists(local_path):
    blob_storage.download_dir(blob_uri)

# Create a CuratedMDDir object to access the data
raw_data = CuratedMDDir(root=Path(local_path))

# Load the trajectory fragments
fragments = [load_simulation_from_files(sim, raw_data.topology_file) for sim in raw_data.simulations]

# Join the fragments into a single trajectory
if fragments:
    traj = mdtraj.join(fragments)
    print(f"Created trajectory with {traj.n_frames} frames")

    # Print some information about the structure
    print(f"Structure information:")
    print(f"  Number of chains: {len(list(traj.topology.chains))}")
    print(f"  Number of residues: {traj.n_residues}")
    print(f"  Number of atoms: {traj.n_atoms}")
    print(f"  Sequence: {traj.topology.to_fasta()}")
else:
    print("No fragments to join into a trajectory")

# %%
# Working with MDTraj Trajectories
# ===============================
#
# Key attributes of mdtraj.Trajectory:
# - traj.xyz: Numpy array with shape (n_frames, n_atoms, 3) containing coordinates
# - traj.time: Numpy array with shape (n_frames) containing time in picoseconds
# - traj.unitcell_lengths: Unit cell lengths for each frame
# - traj.unitcell_angles: Unit cell angles for each frame
# - traj.topology: Topology object containing structural information
#
# The topology object has these useful attributes:
# - traj.topology.n_atoms: Number of atoms
# - traj.topology.n_residues: Number of residues
# - traj.topology.n_chains: Number of chains
# - traj.topology.atoms: Iterator over atom objects
# - traj.topology.residues: Iterator over residue objects
# - traj.topology.chains: Iterator over chain objects

# Example: Extracting specific atom types

# 1. Select all alpha carbon atoms (CA)
ca_indices = [atom.index for atom in traj.topology.atoms if atom.name == 'CA']
ca_traj = traj.atom_slice(ca_indices)
print(f"\nExtracted {ca_traj.n_atoms} alpha carbon atoms")

# 2. Extract CA atoms using MDTraj's DSL
ca_indices = traj.topology.select('name CA')
ca_traj = traj.atom_slice(ca_indices)
print(f"Extracted {ca_traj.n_atoms} alpha carbon atoms")

# %%


caca = np.linalg.vector_norm(ca_traj.xyz[:, :-1] - ca_traj.xyz[:, 1:], axis=-1)

_ = plt.hist(caca.flatten(), bins=100, density=True)

# %%

backbone = [atom.index for atom in traj.topology.atoms if atom.name in ['N', 'CA', 'C', 'O']]

backbone_traj = traj.atom_slice(backbone)

# %%
# Concise version: Extract backbone atoms per residue into [SimulationStep, Residue, backbone Atom, 3] array

backbone_xyz = backbone_traj.xyz
backbone_coords = 10*backbone_xyz.reshape(backbone_xyz.shape[0], -1, 4, 3)

# %%

avg_caca_dist = np.linalg.norm(backbone_coords[:,:-1,1] - backbone_coords[:,1:,1], axis=-1).mean()

print(f"Average CA-CA distance: {avg_caca_dist:.3f} A")
# current C with next N
avg_cn_dist = np.linalg.norm(backbone_coords[:,1:,0] - backbone_coords[:,:-1,2], axis=-1).mean()
print(f"Average C-N distance: {avg_cn_dist:.3f} A")


# Count CA-CA distances larger than 4.5 nm
large_caca_distances = caca > 4.5
num_large_distances = large_caca_distances.sum()
total_distances = caca.size

print(f"CA-CA distances > 4.5 nm: {num_large_distances} out of {total_distances} ({num_large_distances/total_distances*100:.2f}%)")


# %%

import torch
from tqdm.auto import tqdm
from bioemu.steering import potential_loss_fn



# Convert backbone coordinates to torch tensors for batch operations
backbone_coords_torch = torch.from_numpy(backbone_coords).float()

print(f"{backbone_coords_torch.shape=}")
backbone_coords_torch = backbone_coords_torch[::100]
print(f'Subsampled Ca-Ca distances: {torch.linalg.vector_norm(backbone_coords_torch[:,:-1,1]- backbone_coords_torch[:,1:,1], axis=-1).mean()}')

n_residues = backbone_coords_torch.shape[1]
offset = 4
mask = torch.ones(n_residues, n_residues, dtype=torch.bool).triu(diagonal=offset)

ca_coords = backbone_coords_torch[:,:,1]

caca_offset_diag = torch.cdist(ca_coords, ca_coords)[:,mask]


# Plot histogram of all pairwise distances
plt.figure(figsize=(10, 6))
_ = plt.hist(caca_offset_diag.flatten().numpy(), bins=200, density=True, alpha=0.7)
plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)

# Add line for smallest value
min_distance = caca_offset_diag.min().item()
plt.axvline(x=min_distance, color='green', linestyle='--', alpha=0.7, label=f'Min: {min_distance:.2f} Å')

# Plot ReLU function over the histogram
threshold = 4.2
x_range = np.linspace(0, 10, 1000)
potential = np.maximum(0,  0.1*(threshold - x_range))

# Scale ReLU to fit on the same plot as histogram
# relu_scaled = relu_values * plt.ylim()[1] / relu_values.max() * 0.5

plt.plot(x_range, potential, 'r-', linewidth=3, label=f'ReLU(dist - {threshold}) (scaled)', alpha=0.8)
# plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {threshold} Å')


# Add text annotation for the minimum distance
plt.text(min_distance - 0.5, plt.ylim()[1] * 0.1, f'Min: {min_distance:.2f} Å', 
         rotation=0, ha='left', va='center', fontsize=10, 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.xlabel('Distance (A)')
plt.ylabel('Density')
plt.title(f'Distribution of Ca-Ca Pairwise Distances (offset={offset})\noffset={offset}: [i, {"x, " * (offset-1)}i+{offset}, ...]')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(0, 0.1)
plt.legend()
plt.show()



