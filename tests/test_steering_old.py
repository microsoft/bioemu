
# import shutil
# import os
# import wandb
# import pytest
# import torch
# from torch_geometric.data.batch import Batch
# from bioemu.sample import main as sample
# from bioemu.steering import CNDistancePotential, ChainBreakPotential, ChainClashPotential, batch_frames_to_atom37, StructuralViolation
# from pathlib import Path
# import numpy as np
# import random
# import hydra
# from omegaconf import OmegaConf

# # Set fixed seeds for reproducibility
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

# # @pytest.fixture
# # def sequences():
# # 	return [
# # 		'EPVKFKDCGSWVGVIKEVNVSPCPTQPCKLHRGQSYSVNVTFTSNTQSQSSKAVVHGIVMGIPVPFPIPESDGCKSGIRCPIEKDKTYNYVNKLPVKNEYPSIKVVVEWELTDDKNQRFFCWQIPIEVEA',
# # 		'MTHDNKLQVEAIKRGTVIDHIPAQIGFKLLSLFKLTETDQRITIGLNLPSGEMGRKDLIKIENTFLSEDQVDQLALYAPQATVNRIDNYEVVGKSRPSLPERIDNVLVCPNSNCISHAEPVSSSFAVRKRANDIALKCKYCEKEFSHNVVLAN'
# # 	]


# @pytest.mark.parametrize("sequence", ['GYDPETGTWG'], ids=['chignolin'])
# # @pytest.mark.parametrize("FK", [True, False], ids=['FK', 'NoFK'])
# def test_steering(sequence):
#     '''
#     Tests the generation of samples with steering
#     check for sequences: https://github.com/microsoft/bioemu-benchmarks/blob/main/bioemu_benchmarks/assets/multiconf_benchmark_0.1/crypticpocket/testcases.csv
#     '''
#     denoiser_config_path = Path("../bioemu/src/bioemu/config/bioemu.yaml").resolve()

#     # Load config using Hydra in test mode
#     with hydra.initialize_config_dir(config_dir=str(denoiser_config_path.parent), job_name="test_steering"):
#         cfg = hydra.compose(config_name=denoiser_config_path.name,
#                             overrides=[
#                                 f"sequence={sequence}",
#                                 "logging_mode=disabled",
#                                 'steering=chingolin_steering',
#                                 "batch_size_100=200",
#                                 "steering.do_steering=true",
#                                 "steering.num_particles=10",
#                                 "steering.resample_every_n_steps=5",
#                                 "num_samples=1024",])
#     print(OmegaConf.to_yaml(cfg))
#     if cfg.steering.do_steering is False:
#         cfg.steering.num_particles = 1
#     output_dir = f"./outputs/test_steering/FK_{sequence[:10]}_len:{len(sequence)}"
#     output_dir += '_steered' if cfg.steering.do_steering else ''
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)
#     fk_potentials = hydra.utils.instantiate(cfg.steering.potentials)
#     fk_potentials = list(fk_potentials.values())
#     samples: dict = sample(sequence=sequence, num_samples=cfg.num_samples, batch_size_100=cfg.batch_size_100,
#                            output_dir=output_dir,
#                            denoiser_config=cfg.denoiser,
#                            fk_potentials=fk_potentials,
#                            steering_config=cfg.steering)


# # Test Test Tes
