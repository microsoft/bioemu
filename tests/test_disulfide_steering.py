"""
Tests for DisulfideBridgePotential implementation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from bioemu.steering import DisulfideBridgePotential


def test_disulfide_bridge_potential_init():
    """Test initialization of DisulfideBridgePotential."""
    # Test with default parameters
    potential = DisulfideBridgePotential()
    assert potential.flatbottom == 0.01
    assert potential.slope == 10.0
    assert potential.weight == 1.0
    assert potential.specified_pairs == []
    
    # Test with custom parameters
    specified_pairs = [(3, 40), (4, 32), (16, 26)]
    potential = DisulfideBridgePotential(
        flatbottom=0.02,
        slope=5.0,
        weight=2.0,
        specified_pairs=specified_pairs
    )
    assert potential.flatbottom == 0.02
    assert potential.slope == 5.0
    assert potential.weight == 2.0
    assert potential.specified_pairs == specified_pairs


def test_disulfide_bridge_potential_call_no_pairs():
    """Test DisulfideBridgePotential with no specified pairs."""
    potential = DisulfideBridgePotential()
    
    # Create dummy Ca positions
    batch_size = 5
    seq_len = 10
    Ca_pos = torch.rand(batch_size, seq_len, 3) * 10  # Random positions in Angstroms
    
    # Calculate potential energy
    energy = potential(None, Ca_pos, None, None)
    
    # Energy should be zero for all samples when no pairs are specified
    assert energy.shape == (batch_size,)
    assert torch.all(energy == 0)


def test_disulfide_bridge_potential_call_with_pairs():
    """Test DisulfideBridgePotential with specified pairs."""
    # Example sequence from feature plan: TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
    # Disulfide Bridges: [(3,40),(4,32),(16,26)]
    specified_pairs = [(3, 40), (4, 32), (16, 26)]
    potential = DisulfideBridgePotential(specified_pairs=specified_pairs)
    
    batch_size = 3
    seq_len = 45  # Length of the example sequence
    
    # Create a batch of Ca positions where all disulfide bridges are in valid range
    Ca_pos = torch.zeros(batch_size, seq_len, 3)
    
    # Set positions for valid disulfide bridges (distance between 3.75Å and 6.6Å)
    valid_distance = 5.0  # Middle of valid range
    
    # Set positions for each cysteine pair
    for i, j in specified_pairs:
        Ca_pos[:, i, 0] = 0
        Ca_pos[:, j, 0] = valid_distance
    
    # Calculate potential energy
    energy_valid = potential(None, Ca_pos, None, None)
    
    # Energy should be zero for all samples when all bridges are in valid range
    assert torch.allclose(energy_valid, torch.zeros_like(energy_valid))
    
    # Now create a batch where disulfide bridges are outside valid range
    Ca_pos_invalid = Ca_pos.clone()
    
    # Set first bridge too close
    Ca_pos_invalid[0, 3, 0] = 0
    Ca_pos_invalid[0, 40, 0] = 2.0  # Below min_valid_dist (3.75Å)
    
    # Set second bridge too far
    Ca_pos_invalid[1, 4, 0] = 0
    Ca_pos_invalid[1, 32, 0] = 8.0  # Above max_valid_dist (6.6Å)
    
    # Calculate potential energy
    energy_invalid = potential(None, Ca_pos_invalid, None, None)
    
    # Energy should be positive for samples with invalid bridges
    assert energy_invalid[0] > 0
    assert energy_invalid[1] > 0
    assert energy_invalid[2] == 0  # Third sample still has valid bridges


def test_disulfide_bridge_potential_scaling():
    """Test that the potential scales correctly with weight parameter."""
    specified_pairs = [(0, 1)]
    weight = 2.0
    potential = DisulfideBridgePotential(specified_pairs=specified_pairs, weight=weight)
    
    # Create Ca positions with invalid bridge
    batch_size = 1
    seq_len = 2
    Ca_pos = torch.zeros(batch_size, seq_len, 3)
    Ca_pos[0, 1, 0] = 10.0  # Far outside valid range
    
    # Calculate potential energy
    energy = potential(None, Ca_pos, None, None)
    
    # Calculate with weight=1.0 for comparison
    potential_unweighted = DisulfideBridgePotential(specified_pairs=specified_pairs, weight=1.0)
    energy_unweighted = potential_unweighted(None, Ca_pos, None, None)
    
    # Energy should scale with weight
    assert torch.isclose(energy, energy_unweighted * weight)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
