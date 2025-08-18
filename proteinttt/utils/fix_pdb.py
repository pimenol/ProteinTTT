import argparse
from Bio.PDB import PDBParser, PDBIO
import re

def extract_chain_data(structure, chain_id):
    """Extract residue names and numbering for a given chain."""
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [res for res in chain if res.id[0] == ' ']  # skip heteroatoms
                seq = [res.resname for res in residues]
                nums = [f"{res.id[1]}{res.id[2]}" for res in residues]
                return seq, nums
    raise ValueError(f"Chain {chain_id} not found in structure.")


def check_sequence(orig_seq, pred_seq, tag):
    """Check if residue sequences match exactly."""
    if len(orig_seq) != len(pred_seq):
        raise ValueError(f"Length mismatch: original {len(orig_seq)}, predicted {len(pred_seq)}, {tag}")
    mismatches = [(i + 1, o, p) for i, (o, p) in enumerate(zip(orig_seq, pred_seq)) if o != p]
    if mismatches:
        raise ValueError(f"Sequence mismatches:\n" + "\n".join(f"{pos}: {o} != {p}" for pos, o, p in mismatches))
    print("Sequence matches exactly.")


def rename_and_renumber(predicted_structure, target_chain_id, target_numbers):
    """Rename chain and renumber residues to match target numbering."""
    atom_counter = 1
    for model in predicted_structure:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError("Predicted PDB must have exactly one chain.")
        chain = chains[0]
        chain.id = target_chain_id
        print(chain)
        residues = [res for res in chain if res.id[0] == ' ']
        if len(residues) != len(target_numbers):
            raise ValueError("Residue count mismatch after sequence check.")
        for res, num in zip(residues, target_numbers):
            num_str = str(num).strip()
            match = re.match(r'^(\d+)(\D*)$', num_str)
            if match:
                res_num = int(match.group(1))
                ins_code = match.group(2) if match.group(2) else ' '
            else:
                raise ValueError(f"Cannot parse residue numbering: {num}")
            res.id = (res.id, res_num, ins_code)
            for atom in res:
                atom.serial_number = atom_counter
                atom_counter += 1


def fix_pdb(original_pdb, predicted_pdb, chain_id, output_pdb):
    parser = PDBParser(QUIET=True)
    original_pdb = str(original_pdb)

    predicted_pdb = str(predicted_pdb)
    output_pdb = str(output_pdb)

    # Get original chain data
    orig_structure = parser.get_structure("orig", original_pdb)
    orig_seq, orig_nums = extract_chain_data(orig_structure, chain_id)
    # print(orig_nums)

    # Get predicted single chain data
    pred_structure = parser.get_structure("pred", predicted_pdb)
    pred_chains = list(pred_structure.get_chains())
    if len(pred_chains) != 1:
        raise ValueError("Predicted PDB must contain exactly one chain.")
    pred_seq = [res.resname for res in pred_chains[0] if res.id[0] == ' ']

    # Compare sequences
    check_sequence(orig_seq, pred_seq, predicted_pdb)

    # Rename+renumber
    rename_and_renumber(pred_structure, chain_id, orig_nums)

    # Save updated PDB
    io = PDBIO()
    io.set_structure(pred_structure)
    io.save(output_pdb)