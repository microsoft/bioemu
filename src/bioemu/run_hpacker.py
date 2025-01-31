# To be executed in separate python env. See README.md (section "side-chain reconstruction").
try:
    from hpacker import HPacker
    _HAS_HPACKER = True
except Exception:
    _HAS_HPACKER = False



def _hpacker(protein_pdb_in: str, protein_pdb_out: str) -> None:
    """Call hpacker to reconstruct sidechains.

    Args:
        protein_pdb_in: Input PDB file with backbone only.
        protein_pdb_out: Output PDB file with sidechains reconstructed.
    
    """
    if not _HAS_HPACKER:
        raise ImportError("hpacker not found, please install hpacker first")
    hpacker = HPacker(protein_pdb_in)
    hpacker.reconstruct_sidechains(num_refinement_iterations=5)
    hpacker.write_pdb(protein_pdb_out)


if __name__ == "__main__":
    import sys

    _hpacker(sys.argv[1], sys.argv[2])
