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

def fetch_and_parse_sequences(df, pdb_dir, id_column='pdb'):
    parser = PDBParser(QUIET=True)
    output_rows = []

    for _, row in df.iterrows():
        pdb_id = row[id_column]
        pdb_path = pdb_dir / f"{pdb_id}.pdb"

        chain_type_map = {}
        for col in ['Hchain', 'Lchain', 'antigen_chain']:
            if col in df.columns and pd.notna(row[col]):
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
            if chain_id in chain_type_map or id_column != 'pdb':
                seq = ""
                for residue in chain:
                    if residue.id[0] == ' ':
                        resname = residue.get_resname()
                        try:
                            seq += seq1(resname)
                        except KeyError:
                            print(f"Warning: Unknown residue {resname} in chain {chain_id} of PDB {pdb_id}. Skipping.")
                            continue

                output_rows.append({
                    "id": pdb_id,
                    "chain": chain_id,
                    "chain_type": chain_type_map[chain_id] if id_column == 'pdb' else None,
                    "sequence": seq,
                    "resolution": row['resolution'] if id_column == 'pdb' else None
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


from Bio.PDB import PDBList, MMCIFParser, PDBIO, Select
import os


def download_structure_and_extract_chain(pdb_code: str, chain_id: str, output_dir: str = './') -> str:
    """
    Downloads a PDB or mmCIF structure, extracts a specific chain, and saves it to a new PDB file.
    This function handles both modern mmCIF and legacy PDB formats.

    Args:
        pdb_code (str): The 4-character PDB or PDBx/mmCIF identifier.
        chain_id (str): The ID of the chain to extract (e.g., 'A', 'BM').
        output_dir (str): The directory where the output file will be saved. Defaults to the current directory.

    Returns:
        str: The path to the newly created PDB file containing only the specified chain.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Step 1: Download the structure file ---
    pdbl = PDBList()
    # We will try to download in mmCIF format first, as it's the standard for large structures.
    # The 'retrieve_pdb_file' function returns the exact path to the downloaded file.
    try:
        file_path = pdbl.retrieve_pdb_file(pdb_code, pdir=output_dir, file_format='mmCif')
        if not os.path.exists(file_path):
            # Fallback to pdb format if mmCIF fails or doesn't exist
            file_path = pdbl.retrieve_pdb_file(pdb_code, pdir=output_dir, file_format='pdb')
    except Exception as e:
        print(f"Failed to download structure {pdb_code}. Error: {e}")
        return None

    # Check if download was successful
    if not os.path.exists(file_path):
        print(f"Could not download or find the file for PDB code {pdb_code}.")
        return None

    # --- Step 2: Parse the structure and select the chain ---
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_code, file_path)

    # Define a custom class to select only the desired chain
    class ChainSelect(Select):
        def __init__(self, required_chain_id):
            self.required_chain_id = required_chain_id

        def accept_chain(self, chain):
            return chain.id == self.required_chain_id

    # --- Step 3: Write the selected chain to a new PDB file ---
    io = PDBIO()
    io.set_structure(structure)

    output_filename = f"{pdb_code}_{chain_id}.pdb"
    output_path = os.path.join(output_dir, output_filename)

    # Save the structure, using the selector to filter for the specific chain
    io.save(output_path, ChainSelect(chain_id))

    print(f"Successfully extracted chain {chain_id} from {pdb_code} and saved to {output_path}")

    # --- Step 4: Clean up the downloaded source file and directory ---
    try:
        os.remove(file_path)
        # Attempt to remove the directory PDBList creates (e.g., 'yf/' for '6yfg')
        # This will only succeed if the directory is empty.
        os.rmdir(os.path.dirname(file_path))
    except OSError:
        # This is expected if the directory is not empty or if multiple files were downloaded.
        pass

    return output_path


import requests

def download_chain_cif_to_pdb(pdb_id, chain_id, output_file):
    """
    Downloads a specific chain from a PDB entry in CIF format and saves it as a PDB file.

    Args:
        pdb_id (str): The 4-character PDB ID (e.g., "1TND").
        chain_id (str): The single-character chain ID to extract (e.g., "A").
        output_file (str): The path to the output PDB file (e.g., "chainA.pdb").
    """
    # Construct the download URL for the CIF file
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CIF file for PDB ID {pdb_id}: {e}")
        return

    cif_lines = response.text.splitlines()

    # Find the start of the atom_site data loop
    in_atom_site_loop = False
    headers = []
    data_lines = []
    
    for line in cif_lines:
        if line.startswith('_atom_site.'):
            headers.append(line.strip())
        elif line.strip() == 'loop_':
            # A new loop starts, if we have headers, the next lines are data
            if headers:
                in_atom_site_loop = True
        elif in_atom_site_loop:
            if line.startswith('#') or not line.strip():
                # End of the loop data
                break
            data_lines.append(line)

    if not headers or not data_lines:
        print("Could not find _atom_site data in the CIF file.")
        return

    # Map header names to their index
    try:
        header_map = {h.split('.')[1]: i for i, h in enumerate(headers)}
        group_pdb_idx = header_map['group_PDB']
        chain_idx = header_map['auth_asym_id'] # Use auth_asym_id for author-assigned chain
        atom_idx = header_map['label_atom_id']
        res_name_idx = header_map['label_comp_id']
        res_seq_idx = header_map['auth_seq_id']
        x_idx = header_map['Cartn_x']
        y_idx = header_map['Cartn_y']
        z_idx = header_map['Cartn_z']
        occupancy_idx = header_map['occupancy']
        bfactor_idx = header_map['B_iso_or_equiv']
        element_idx = header_map['type_symbol']
    except KeyError as e:
        print(f"Missing expected CIF column: {e}")
        return

    # Process and filter atom lines
    pdb_lines = []
    atom_serial = 1
    for line in data_lines:
        # CIF fields can be quoted if they contain spaces
        fields = []
        in_quote = False
        current_field = ''
        for char in line:
            if char == "'" or char == '"':
                in_quote = not in_quote
            elif char.isspace() and not in_quote:
                if current_field:
                    fields.append(current_field)
                    current_field = ''
            else:
                current_field += char
        if current_field:
            fields.append(current_field)

        if fields[chain_idx] == chain_id:
            record_type = fields[group_pdb_idx]
            atom_name = fields[atom_idx]
            res_name = fields[res_name_idx]
            res_seq = fields[res_seq_idx]
            x, y, z = float(fields[x_idx]), float(fields[y_idx]), float(fields[z_idx])
            occupancy = float(fields[occupancy_idx])
            temp_factor = float(fields[bfactor_idx])
            element = fields[element_idx]

            # Format atom name to be 4 characters, left-aligned if short
            if len(atom_name) < 4 and len(element) == 1:
                formatted_atom_name = f" {atom_name:<3}"
            else:
                formatted_atom_name = f"{atom_name:^4}"

            # Format to PDB ATOM record specification
            pdb_line = (f"{record_type:<6}{atom_serial:>5} {formatted_atom_name:<4}{res_name:>3} "
                        f"{chain_id:>1}{res_seq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{occupancy:6.2f}{temp_factor:6.2f}          {element:>2}")
            pdb_lines.append(pdb_line)
            atom_serial += 1

    # Write the collected PDB lines to the output file
    try:
        with open(output_file, 'w') as f:
            f.write(f"HEADER    CHAIN {chain_id} FROM PDB ID {pdb_id.upper()}\n")
            for line in pdb_lines:
                f.write(line + '\n')
            f.write("END\n")
        print(f"Successfully saved chain {chain_id} from {pdb_id} to {output_file}")
    except IOError as e:
        print(f"Error writing to file {output_file}: {e}")

