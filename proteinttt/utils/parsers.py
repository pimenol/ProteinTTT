from Bio.SeqUtils import seq1
import pandas as pd
from Bio.PDB import PDBParser


def calculate_plddt_by_CDRs(pdb_path, is_heavy):
    pLDDT_scores = {
        'H1': [],
        'H2': [],
        'H3': [],
        'L1': [],
        'L2': [],
        'L3': []
    }

    if is_heavy == True:
        CDR_RANGES = {
            'H1': (26, 32),
            'H2': (52, 56),
            'H3': (95, 102),
        }
    elif is_heavy == False:
        CDR_RANGES = {
            'L1': (24, 34),
            'L2': (50, 56),
            'L3': (89, 97)
        }
    else:
        raise ValueError(f"{pdb_path}: is_heavy: {is_heavy}. Chain is antigen.")

    def get_cdr_name(residue_number: int):
        """Return CDR name if residue is in CDR region, else None."""
        for cdr_name, (start, end) in CDR_RANGES.items():
            if start <= residue_number <= end:
                return cdr_name
        return None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # Parse residue number
                residue_num_str = line[22:26].strip()
                try:
                    if residue_num_str[-1].isalpha():
                        residue_num = int(residue_num_str[:-1])
                    else:
                        residue_num = int(residue_num_str)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

                # Parse pLDDT score
                try:
                    plddt_score = float(line[60:66].strip())
                except ValueError:
                    print(f"Skipping line due to ValueError in pLDDT parsing: {line.strip()}")
                    continue

                cdr_name = get_cdr_name(residue_num)
                if cdr_name:
                    pLDDT_scores[cdr_name].append(plddt_score)

    # Calculate average pLDDT for each CDR
    avg_pLDDT = {}
    for cdr_name, scores in pLDDT_scores.items():
        if scores:
            avg_pLDDT[cdr_name] = sum(scores) / len(scores)
        else:
            avg_pLDDT[cdr_name] = None
    avg_pLDDT['CDRs'] = sum([score for score in avg_pLDDT.values() if score is not None]) / len([score for score in avg_pLDDT.values() if score is not None])
    return avg_pLDDT

def extract_cdr_to_new_pdb(input_pdb_path, output_pdb_path, chain, is_heavy):
    if is_heavy == True:
        CDR_RANGES = {
            'H1': (26, 32),
            'H2': (52, 56),
            'H3': (95, 102),
        }
    elif is_heavy == False:
        CDR_RANGES = {
            'L1': (24, 34),
            'L2': (50, 56),
            'L3': (89, 97)
        }
    else:
        raise ValueError(f"{input_pdb_path}: is_heavy: {is_heavy}. Chain is antigen.")

    def is_in_cdr_region(residue_number: int):
        """Check if residue is in CDR region."""
        for start, end in CDR_RANGES.values():
            if start <= residue_number <= end:
                return True
        return False

    cdr_lines = []
    with open(input_pdb_path, 'r') as f:
        for line in f:
            if (line.startswith('ATOM')) and line[21].strip() == chain:
                # Parse residue number
                residue_num_str = line[22:26].strip()
                try:
                    if residue_num_str[-1].isalpha():
                        residue_num = int(residue_num_str[:-1])
                    else:
                        residue_num = int(residue_num_str)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

                # Check if it's in CDR and is a CA atom (can be adjusted)
                if is_in_cdr_region(residue_num):
                    # print(line)
                    cdr_lines.append(line)

    # Write the collected lines to the output PDB
    with open(output_pdb_path, 'w') as f:
        for line in cdr_lines:
            f.write(line)

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
                    # if residue.has_id('CA'):
                    if residue.id[0] == ' ':
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



def extract_cdr_to_new_pdb(input_pdb_path, output_pdb_path, chain, is_heavy):
    if is_heavy == True:
        CDR_RANGES = {
            'H1': (26, 32),
            'H2': (52, 56),
            'H3': (95, 102),
        }
    elif is_heavy == False:
        CDR_RANGES = {
            'L1': (24, 34),
            'L2': (50, 56),
            'L3': (89, 97)
        }
    else:
        raise ValueError(f"{input_pdb_path}: is_heavy: {is_heavy}. Chain is antigen.")

    def is_in_cdr_region(residue_number: int):
        """Check if residue is in CDR region."""
        for start, end in CDR_RANGES.values():
            if start <= residue_number <= end:
                return True
        return False

    cdr_lines = []
    with open(input_pdb_path, 'r') as f:
        for line in f:
            if (line.startswith('ATOM')) and line[21].strip() == chain:
                # Parse residue number
                residue_num_str = line[22:26].strip()
                try:
                    if residue_num_str[-1].isalpha():
                        residue_num = int(residue_num_str[:-1])
                    else:
                        residue_num = int(residue_num_str)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

                # Check if it's in CDR and is a CA atom (can be adjusted)
                if is_in_cdr_region(residue_num):
                    # print(line)
                    cdr_lines.append(line)

    # Write the collected lines to the output PDB
    with open(output_pdb_path, 'w') as f:
        for line in cdr_lines:
            f.write(line)

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
                    # if residue.has_id('CA'):
                    if residue.id[0] == ' ':
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


def extract_h3_to_new_pdb(input_pdb_path, output_pdb_path, chain, is_heavy):
    if is_heavy == True:
        CDR_RANGES = {
            'H3': (95, 102),
        }
    else:
        return

    def is_in_cdr_region(residue_number: int):
        """Check if residue is in CDR region."""
        for start, end in CDR_RANGES.values():
            if start <= residue_number <= end:
                return True
        return False

    cdr_lines = []
    with open(input_pdb_path, 'r') as f:
        for line in f:
            if (line.startswith('ATOM')) and line[21].strip() == chain:
                # Parse residue number
                residue_num_str = line[22:26].strip()
                try:
                    if residue_num_str[-1].isalpha():
                        residue_num = int(residue_num_str[:-1])
                    else:
                        residue_num = int(residue_num_str)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
                    continue

                if is_in_cdr_region(residue_num):
                    cdr_lines.append(line)
    
    if len(cdr_lines) <= 0:
        # print(f"No CDR lines found in {input_pdb_path} for chain {chain}. Skipping. ")
        return
    # Write the collected lines to the output PDB
    with open(output_pdb_path, 'w') as f:
        for line in cdr_lines:
            f.write(line)
