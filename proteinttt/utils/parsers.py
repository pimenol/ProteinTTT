from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import pandas as pd


def fetch_and_parse_sequences(df, pdb_dir):
    parser = PDBParser(QUIET=True)
    output_rows = []

    for _, row in df.iterrows():
        pdb_id = row['pdb'].lower()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"

        chain_type_map = {}
        for col in ['Hchain', 'Lchain', 'antigen_chain']:
            if pd.notna(row[col]):
                chains = str(row[col]).replace(" ", "").split('|')
                for ch in chains:
                    chain_type_map[ch] = col

        try:
            structure = parser.get_structure(pdb_id, pdb_path)
        except Exception as e:
            print(f"Error parsing PDB {pdb_id}: {e}")
            continue

        model = structure[0]

        for chain in model:
            chain_id = chain.get_id()
            if chain_id in chain_type_map:
                seq = ""
                for residue in chain:
                    if residue.has_id('CA'):
                        resname = residue.get_resname()
                        try:
                            seq += seq1(resname)
                        except KeyError:
                            print(f"Warning: Unknown residue {resname} in chain {chain_id} of PDB {pdb_id}. Skipping.")
                            continue

                output_rows.append({
                    "pdb": pdb_id,
                    "chain": chain_id,
                    "chain_type": chain_type_map[chain_id],
                    "sequence": seq,
                    "resolution": row['resolution']
                })

    return pd.DataFrame(output_rows)
