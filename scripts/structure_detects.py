import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class _ProteinBackbone:
  atom_positions: np.ndarray  # [num_res, 3, 3] for N, CA, C
  atom_mask: np.ndarray       # [num_res, 3]
  chain_index: np.ndarray     # [num_res]
  b_factors: np.ndarray       # [num_res, 3]


def _parse_pdb_backbone(pdb_str: str) -> '_ProteinBackbone':
  """Parses backbone atoms (N, CA, C) from a PDB string.

  Avoids the alphafold dependency by reading ATOM records directly.
  """
  BACKBONE = {'N': 0, 'CA': 1, 'C': 2}
  residues: Dict = {}  # (chain_id, res_num) -> {atom_name: ([x,y,z], b)}
  chain_order: Dict[str, int] = {}

  for line in pdb_str.splitlines():
    if not line.startswith('ATOM'):
      continue
    alt_loc = line[16]
    if alt_loc not in (' ', 'A'):
      continue
    atom_name = line[12:16].strip()
    if atom_name not in BACKBONE:
      continue
    chain_id = line[21]
    res_num = int(line[22:26])
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    b = float(line[60:66])

    if chain_id not in chain_order:
      chain_order[chain_id] = len(chain_order)
    key = (chain_id, res_num)
    if key not in residues:
      residues[key] = {}
    residues[key][atom_name] = (np.array([x, y, z]), b)

  sorted_keys = sorted(residues, key=lambda k: (chain_order[k[0]], k[1]))
  num_res = len(sorted_keys)
  atom_positions = np.zeros((num_res, 3, 3))
  atom_mask = np.zeros((num_res, 3))
  chain_index = np.zeros(num_res, dtype=int)
  b_factors = np.zeros((num_res, 3))

  for i, key in enumerate(sorted_keys):
    chain_index[i] = chain_order[key[0]]
    for name, idx in BACKBONE.items():
      if name in residues[key]:
        xyz, b = residues[key][name]
        atom_positions[i, idx] = xyz
        atom_mask[i, idx] = 1.0
        b_factors[i, idx] = b

  return _ProteinBackbone(atom_positions, atom_mask, chain_index, b_factors)


def _dihedral_angle(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> np.ndarray:
  """Computes dihedral angles in degrees for arrays of shape [N, 3]."""
  b0 = p0 - p1
  b1 = p2 - p1
  b2 = p3 - p2
  n1 = np.cross(b0, b1)
  n2 = np.cross(b1, b2)
  b1_norm = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
  m1 = np.cross(n1, b1_norm)
  x = np.sum(n1 * n2, axis=-1)
  y = np.sum(m1 * n2, axis=-1)
  return np.degrees(np.arctan2(y, x))


def _compute_phi_psi(
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    chain_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes phi/psi dihedral angles from backbone atom positions.

  Args:
    atom_positions: [num_res, num_atom_type, 3] atom coordinates.
    atom_mask: [num_res, num_atom_type] binary mask for atom presence.
    chain_index: [num_res] chain assignment per residue.

  Returns:
    phi: [num_res] phi angles in degrees (NaN where undefined).
    psi: [num_res] psi angles in degrees (NaN where undefined).
  """
  num_res = atom_positions.shape[0]
  # Backbone atoms: N=0, CA=1, C=2
  n_pos = atom_positions[:, 0]
  ca_pos = atom_positions[:, 1]
  c_pos = atom_positions[:, 2]
  backbone_present = atom_mask[:, 0] * atom_mask[:, 1] * atom_mask[:, 2]

  phi = np.full(num_res, np.nan)
  psi = np.full(num_res, np.nan)

  # Phi[i] = dihedral(C[i-1], N[i], CA[i], C[i]) for i >= 1, same chain
  same_chain_prev = chain_index[1:] == chain_index[:-1]
  valid_phi = same_chain_prev & (backbone_present[:-1] > 0) & (backbone_present[1:] > 0)
  if np.any(valid_phi):
    phi[1:][valid_phi] = _dihedral_angle(
        c_pos[:-1][valid_phi], n_pos[1:][valid_phi],
        ca_pos[1:][valid_phi], c_pos[1:][valid_phi])

  # Psi[i] = dihedral(N[i], CA[i], C[i], N[i+1]) for i < num_res-1, same chain
  same_chain_next = chain_index[:-1] == chain_index[1:]
  valid_psi = same_chain_next & (backbone_present[:-1] > 0) & (backbone_present[1:] > 0)
  if np.any(valid_psi):
    psi[:-1][valid_psi] = _dihedral_angle(
        n_pos[:-1][valid_psi], ca_pos[:-1][valid_psi],
        c_pos[:-1][valid_psi], n_pos[1:][valid_psi])

  return phi, psi


def _classify_secondary_structure(
    phi: np.ndarray, psi: np.ndarray
) -> np.ndarray:
  """Classifies residues as helix (0), sheet (1), or loop (2) from phi/psi.

  Args:
    phi: [num_res] phi angles in degrees (NaN where undefined).
    psi: [num_res] psi angles in degrees (NaN where undefined).

  Returns:
    ss: [num_res] integer array: 0=helix, 1=sheet, 2=loop.
  """
  ss = np.full(len(phi), 2, dtype=int)  # default: loop

  helix = (phi >= -160) & (phi <= -20) & (psi >= -80) & (psi <= 50)
  sheet = ((phi >= -180) & (phi <= -40) &
           (((psi >= 50) & (psi <= 180)) | ((psi >= -180) & (psi <= -90))))

  ss[helix] = 0
  ss[sheet] = 1
  return ss


def is_dynamic_protein(pdb_path: str) -> bool:
  """Returns True if the protein is dynamic (hard to predict).

  A protein is considered dynamic if both conditions hold:
    1. Mean pLDDT < 50 (from B-factor column of AlphaFold PDB output)
    2. More than 50% of residues are loops (not helix or sheet)

  Args:
    pdb_path: Path to a PDB file (AlphaFold output with pLDDT as B-factors).

  Returns:
    True if the protein is dynamic, False otherwise.
  """
  with open(pdb_path, 'r') as f:
    pdb_str = f.read()
  prot = _parse_pdb_backbone(pdb_str)

  # pLDDT is stored as B-factor; use CA atom (index 1) per residue.
  plddt_per_residue = prot.b_factors[:, 1]
  mean_plddt = np.mean(plddt_per_residue)

  phi, psi = _compute_phi_psi(prot.atom_positions, prot.atom_mask,
                               prot.chain_index)
  ss = _classify_secondary_structure(phi, psi)
  loop_fraction = np.sum(ss == 2) / len(ss)

  return bool(mean_plddt < 50 and loop_fraction > 0.5)


def is_one_helix_protein(pdb_path: str) -> bool:
  """Returns True if the protein is a single helix (easy to predict).

  A protein is considered a single helix if both conditions hold:
    1. Mean pLDDT > 80
    2. More than 80% of residues are helices (not sheet or loop)
  """
  with open(pdb_path, 'r') as f:
    pdb_str = f.read()
  prot = _parse_pdb_backbone(pdb_str)

  plddt_per_residue = prot.b_factors[:, 1]
  mean_plddt = np.mean(plddt_per_residue)

  phi, psi = _compute_phi_psi(prot.atom_positions, prot.atom_mask,
                               prot.chain_index)
  ss = _classify_secondary_structure(phi, psi)
  helix_fraction = np.sum(ss == 0) / len(ss)

  return bool(mean_plddt > 80 and helix_fraction > 0.8)

def describe_protein_structure(pdb_path: str) -> str:
  """Describes the protein structure from a PDB file.

  Args:
    pdb_path: Path to a PDB file.

  Returns:
    A string describing the protein structure.
  """
  with open(pdb_path, 'r') as f:
    pdb_str = f.read()
  prot = _parse_pdb_backbone(pdb_str)

  phi, psi = _compute_phi_psi(prot.atom_positions, prot.atom_mask,
                               prot.chain_index)
  ss = _classify_secondary_structure(phi, psi)
  return ss