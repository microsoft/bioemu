
import shutil
import os
import pytest
import torch
from torch_geometric.data.batch import Batch
from bioemu.sample import main as sample
from bioemu.steering import CNDistancePotential, CaCaDistancePotential, CaClashPotential, batch_frames_to_atom37
from pathlib import Path
import numpy as np

# @pytest.fixture
# def sequences():
# 	return [
# 		'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
# 		'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
# 	]


@pytest.mark.parametrize("sequence", [
    'GYDPETGTWG',
    'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
    'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
], ids=['GYDPETGTWG', "EPVKFKDC...", "MTHDNKLQ..."])
def test_generate_batch(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    output_dir = f"./outputs/test_steering/{sequence[:10]}"
    denoiser_config_path = Path("../bioemu/src/bioemu/config/denoiser/dpm.yaml").resolve()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    sample(sequence=sequence, num_samples=10, output_dir=output_dir, denoiser_config_path=denoiser_config_path)


def test_potentials():

    expected = np.load("tests/expected.npz")
    batch = Batch(**expected)
    batch.pos = torch.from_numpy(batch.pos)  # .unsqueeze(0).repeat(2, 1, 1)
    batch.node_orientations = torch.from_numpy(batch.node_orientations)  # .unsqueeze(0)  # .repeat(2, 1, 1)
    batch.sequences = 'GYDPETGTWG'
    caca_potential = CaCaDistancePotential()
    caclash_potential = CaClashPotential()
    caca_potential(batch.pos, batch.node_orientations, batch.sequences)
    atom37, _, _ = batch_frames_to_atom37(batch.pos, batch.node_orientations, batch.sequences)
    cn_bondlength_loss = CNDistancePotential()(atom37)
