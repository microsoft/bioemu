
import shutil
import os
import pytest
import torch
from torch_geometric.data.batch import Batch
from bioemu.sample import main as sample
from bioemu.steering import CNDistancePotential, CaCaDistancePotential, CaClashPotential, batch_frames_to_atom37, StructuralViolation
from pathlib import Path
import numpy as np
import random

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
def test_generate_fk_batch(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    output_dir = f"./outputs/test_steering/{sequence[:10]}"
    denoiser_config_path = Path("../bioemu/src/bioemu/config/denoiser/dpm.yaml").resolve()
    fk_potentials = [StructuralViolation()]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    sample(sequence=sequence, num_samples=10, output_dir=output_dir, denoiser_config_path=denoiser_config_path,
           fk_potentials=fk_potentials, num_fk_samples=3)


@pytest.mark.parametrize("sequence", [
    'QGGTWRLLGSVVFIHRQELITTLYIGFLGLIFSSYFVYLAEKDAVNESGRVEFGSYADALWWGVVTVTTIGYGDKVPQTWVGKTIASCFSVFAISFFALPAGILGSGFALKVQQKQRQKHFNRQIPAAASLIQTAWRCYAAENPDSSTWKIYIRKAPRSHTLLSPSPKPKKSVVVKKKKFKLDKDNGVTPGEKMLTVPHITCDPPEERRLDHFSVDGYDSSVRKSPTLLEVSMPHFMRTNSFAEDLDLEGETLLTPITHISQLREHHRATIKVIRRMQYFVAKKKFQQARKPYDVRDVIEQYSQGHLNLMVRIKELQRRLDQSIGKPSLFISVSEKSKDRGSNTIGARLNRVEDKVTQLDQRLALITDMLHQLLSLHGGSTPGSGGPPREGGAHITQPCGSGGSVDPELFLPSNTLPTYEQLTVPRRGPDEGSLEGGSSGGWSHPQFEK',
    'QGGTWRLLGSVVFIHRQELITTLYIGFLGLIFSSYFVYLAEKDAVNESGRVEFGSYADALWWGVVTVTTIGYGDKVPQTWVGKTIASCFSVFAISFFALPAGILGSGFALKVQQKQRQKHFNRQIPAA',
], ids=['long seq', 'shortened seq'])
# @pytest.mark.parametrize("FK", [True, False], ids=['FK', 'NoFK'])
def test_steering(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    denoiser_config_path = Path("../bioemu/src/bioemu/config/denoiser/dpm.yaml").resolve()

    # FK Steering
    print('Resampling')
    output_dir_FK = f"./outputs/test_steering/FK_{sequence[:10]}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    fk_potentials = [CaCaDistancePotential(), CaClashPotential()]
    sample(sequence=sequence, num_samples=20, output_dir=output_dir_FK, denoiser_config_path=denoiser_config_path,
           fk_potentials=fk_potentials, num_fk_samples=3)

    # No FK Steering
    output_dir_FK = f"./outputs/test_steering/NoFK_{sequence[:10]}"
    if os.path.exists(output_dir_FK):
        shutil.rmtree(output_dir_FK)
    print('\n No FK Steering')
    sample(sequence=sequence, num_samples=20, output_dir=output_dir_FK, denoiser_config_path=denoiser_config_path,
           fk_potentials=fk_potentials, num_fk_samples=None)
