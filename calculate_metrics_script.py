import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import time
import esm
import biotite.structure.io as bsio
from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG
from proteinttt.utils.structure import calculate_tm_score, lddt_score
import torch
import argparse
import os
import uuid
import traceback
import tqdm


def main(start, end):
    # --- Configuration ---
    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    OUT_DIR_ESM = base_path / 'esm_fold'
    SUMMARY_PATH = base_path / f"to_process.tsv"
    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
    
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))

    SAVE_PATH = base_path / 'res_eval' / f"results_with_esmfold_{JOB_SUFFIX}.tsv"
    print(f"SAVE_PATH: {SAVE_PATH}")

    # --- Load Data ---
    df = pd.read_csv(SUMMARY_PATH, sep="\t")

    def calculate_metrics(true_path, pred_path):
        tm_score = calculate_tm_score(pred_path, true_path)
        lddt = lddt_score(true_path, pred_path)

        return tm_score, lddt

    processed_count = 0
    print(f"{SUMMARY_PATH}")
    for i, row in df.iterrows():
        if not (start <= i < end):
            continue
        seq_id = str(row.get("id"))

        # BEFORE ------------------------------
        if (OUT_DIR_ESM / f"{seq_id}.pdb").exists():
            try:
                tm_score_before, lddt_before = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUT_DIR_ESM / f"{seq_id}.pdb"
                )
            except Exception as e:
                warnings.warn(f"Error calculating metrics in already processed: {seq_id}: {e}")
                traceback.print_exc()

            pLDDT_before = float(np.asarray(bsio.load_structure(OUT_DIR_ESM / f"{seq_id}.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())

            df.at[i, 'pLDDT_before'] = pLDDT_before
            df.at[i, 'tm_score_before'] = tm_score_before
            df.at[i, 'lddt_before'] = lddt_before
            # print(f"Metrics calculated for {seq_id} before pLDDT: {df.at[i, 'pLDDT_before']}, after pLDDT: {df.at[i, 'pLDDT_after']}")
            processed_count += 1
        else:
            warnings.warn(f"Sequence {seq_id} not processed before, skipping metrics calculation.")
            continue
        
        if processed_count > 0 and processed_count % 1000 == 0:
            df.to_csv(path_or_buf=SAVE_PATH, sep="\t", index=False)
            # print(f"Saved progress. Total time elapsed: seconds for {processed_count} sequences.")

    # --- Final Save ---
    df.to_csv(SAVE_PATH, sep="\t", index=False)
    print("Final results saved.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run ESMFold and ProteinTTT on a chunk of sequences.")
    parser.add_argument('--chunk_start', type=int, required=True, help='Starting index of the sequence chunk.')
    parser.add_argument('--chunk_end', type=int, required=True, help='Ending index of the sequence chunk.')
    args = parser.parse_args()
    
    main(args.chunk_start, args.chunk_end)
