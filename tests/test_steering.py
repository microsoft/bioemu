from bioemu.sample import main as sample
import shutil
import os
import pytest


# @pytest.fixture
# def sequences():
# 	return [
# 		'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
# 		'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
# 	]


@pytest.mark.parametrize("sequence", [
    'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
    'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
], ids=["sequence1_EPVKFKDC", "sequence2_MTHDNKLQ"])
def test_generate_batch(sequence):
    '''
    Tests the generation of samples with steering
    check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
    '''
    output_dir = f"./outputs/test_steering/{sequence[:10]}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    sample(sequence=sequence, num_samples=10, output_dir=output_dir)
