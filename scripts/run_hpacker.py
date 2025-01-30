# to be executed in separate python env, compare README.md (section "side-chain reconstruction")
from hpacker import HPacker


def _hpacker(protein_pdb_in: str, protein_pdb_out: str) -> None:
    """call to hpacker"""
    hpacker = HPacker(protein_pdb_in)
    hpacker.reconstruct_sidechains(num_refinement_iterations=5)
    hpacker.write_pdb(protein_pdb_out)


if __name__ == "__main__":
    import sys

    _hpacker(sys.argv[1], sys.argv[2])
