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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("protein_pdb_in", help="Input PDB file with backbone only.")
    parser.add_argument("protein_pdb_out", help="Output PDB file with sidechains reconstructed.")
    args = parser.parse_args()
    _hpacker(args.protein_pdb_in, args.protein_pdb_out)
