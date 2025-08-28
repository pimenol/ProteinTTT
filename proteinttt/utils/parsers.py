from Bio.SeqUtils import seq1
import pandas as pd


def fetch_and_parse_sequences(df: pd.DataFrame, pdb_dir):
    """
    Strict, case-sensitive PDB parser for sequences with Chothia numbering.
    """
    output_rows = []

    def one_letter(resname: str) -> str:
        try:
            return seq1(resname)
        except KeyError:
            if resname == "MSE":
                return "M"
            raise

    for _, row in df.iterrows():
        pdb_id = str(row["pdb"]).lower()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"

        chain_type_map = {}
        for col in ["Hchain", "Lchain", "antigen_chain"]:
            if col in row and pd.notna(row[col]):
                for ch in str(row[col]).replace(" ", "").split("|"):
                    if ch:
                        chain_type_map[ch] = col

        try:
            with open(pdb_path, "r") as fh:
                lines = fh.readlines()
        except Exception as e:
            print(f"Error reading PDB {pdb_id}: {e}")
            continue
        
        chains = {}
        seen = {}  

        for line in lines:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16]
            resname = line[17:20].strip()
            chain_id = line[21] 
            resseq_str = line[22:26]
            icode = line[26]

            if atom_name != "CA" or altloc not in (" ", "A"):
                continue

            try:
                resseq = int(resseq_str)
            except ValueError:
                continue

            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen[key] = True

            chains.setdefault(chain_id, []).append((resseq, icode, resname))

        for chain_id, chain_type in chain_type_map.items():
            if chain_id not in chains:
                print(f"Warning: Chain '{chain_id}' not found in PDB {pdb_id}.")
                continue

            residues = chains[chain_id]
            # The file is already in order; if needed, sort by (resseq, icode) safely:
            residues.sort(key=lambda t: (t[0], (t[1] or " ")))

            seq_chars = []
            chothia_numbers = []
            chothia_positions = []

            prefix = "H" if chain_type == "Hchain" else ("L" if chain_type == "Lchain" else "")

            for resseq, icode, resname in residues:
                try:
                    one = one_letter(resname)
                except KeyError:
                    print(f"Warning: Unknown residue {resname} in chain {chain_id} of PDB {pdb_id}. Skipping.")
                    continue

                seq_chars.append(one)
                num = f"{resseq}{icode.strip()}" if icode.strip() else str(resseq)
                chothia_numbers.append(num)
                chothia_positions.append(f"{prefix}{num}" if prefix else num)

            output_rows.append({
                "pdb": pdb_id,
                "chain": chain_id,
                "chain_type": chain_type,
                "sequence": "".join(seq_chars),
                "chothia_numbers": chothia_numbers,
                "chothia_positions": chothia_positions,
                "resolution": row.get("resolution", None),
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
                if is_in_cdr_region(residue_num) and line[12:16].strip() == 'CA':
                    # print(line)
                    cdr_lines.append(line)

    # Write the collected lines to the output PDB
    with open(output_pdb_path, 'w') as f:
        for line in cdr_lines:
            f.write(line)
