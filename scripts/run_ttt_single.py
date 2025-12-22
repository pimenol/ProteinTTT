#!/scratch/project/open-35-8/pimenol1/miniconda3/envs/proteinttt2/bin/python
#SBATCH --job-name=ttt_single
#SBATCH --account=OPEN-35-8
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=./ttt_single_%A.out
#SBATCH --error=./ttt_single_%A.err

# sbatch scripts/run_ttt_single.py /scratch/project/open-35-8/data/cameo/proteinttt_test/7qii_B.pdb /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/scripts
# sbatch scripts/run_ttt_single.sh /scratch/project/open-35-8/data/cameo/proteinttt_test/7qii_B.pdb /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/scripts

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, "/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT")

import argparse
import torch
import esm

from proteinttt.models.esmfold import ESMFoldTTT, GRAD_CLIP_ESMFOLD_TTT_CFG, DEFAULT_ESMFOLD_TTT_CFG
from proteinttt.utils.structure import get_sequence_from_pdb


def main():
    parser = argparse.ArgumentParser(description="Run ProteinTTT on a single PDB file")
    # parser.add_argument("pdb_path", type=str, help="Path to the PDB file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for CSV results (default: same directory as input PDB)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output CSV filename (default: <pdb_stem>_ttt.csv)",
    )
    args = parser.parse_args()

    # pdb_path = Path(args.pdb_path)
    # if not pdb_path.exists():
    #     raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Set output path
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("./")

    if args.output_name:
        out_csv = out_dir / args.output_name
    else:
        out_csv = out_dir / f"ttt_single.csv"

    # Load model
    print("Loading ESMFold model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    base_model = esm.pretrained.esmfold_v1().eval().to(device)
    ttt_cfg = GRAD_CLIP_ESMFOLD_TTT_CFG
    # ttt_cfg.score_seq_kind = "scaled_pseudo_perplexity"
    model = ESMFoldTTT.ttt_from_pretrained(
        base_model,
        ttt_cfg=ttt_cfg,
        esmfold_config=base_model.cfg,
    ).to(device)
    model.set_chunk_size(128)

    # Extract sequence
    sequence = "MTRDWFNQICEEIKYQEKPIEISKNGHEPVSLRRLFKESFSCLVEERWIWDLYYRPLRMREFWDESRAEKYFYATTFSVFILKYSYGSDISTTSTRHLHRHTVLTCYLDSIVDMGWLSWAKQFGMAVFGLIPDIPPFDPNQPVGFRSSIEEDFPRLRLLAEGPHKMEIMKSLLEAAAVEKTRDSATSFQDIARYRMESNLVCIRPFIPAMGDLLQPLAMMYSFFDDAMDVIEDVDAGQPSYLTNNEDIRRGGNIARAAMSELNSLSKVDWSWLAQAAILLSEVNAMLQISLSINEYNNIENIGKHLFVRIAVLFFIILQ"
    # print(f"PDB file: {pdb_path.name}")
    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")

    # Run TTT
    print("Running TTT...")
    model.ttt_reset()
    df_ttt = model.ttt(sequence)["df"]

    # Save result
    df_ttt.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open("result.pdb", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()

