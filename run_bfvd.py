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


def main(start, end, date, calculate_only_ttt=False):
    # --- Configuration ---
    CALCULATE_ONLY_TTT = calculate_only_ttt
    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    OUT_DIR = base_path / 'predicted_structures'
    SUMMARY_PATH = base_path / 'subset_1.tsv' if not CALCULATE_ONLY_TTT else base_path / 'to_process.tsv'
    LOGS_DIR = base_path / 'logs'

    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
    
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))
    SAVE_PATH = base_path /f"results_ttt_{start}_{end}_{JOB_SUFFIX}.tsv"  if CALCULATE_ONLY_TTT else base_path / f"results_1_{start}_{end}_{JOB_SUFFIX}.tsv"

    print(f"SAVE_PATH: {SAVE_PATH}")

    # --- Load Data ---
    df = pd.read_csv(SUMMARY_PATH, sep="\t")

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = esm.pretrained.esmfold_v1().eval().to(device)
    ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    ttt_cfg.steps = 20
    ttt_cfg.seed = 0
    model = ESMFoldTTT.ttt_from_pretrained(
        base_model,
        ttt_cfg=ttt_cfg,
        esmfold_config=base_model.cfg
    ).to(device)

    def predict_structure(model, sequence, pdb_id, tag, out_dir=OUT_DIR):
        with torch.no_grad():
            pdb_str = model.infer_pdb(sequence)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        out_path = out_dir / f"{pdb_id}{tag}.pdb"
        out_path.write_text(pdb_str)

        struct = bsio.load_structure(out_path, extra_fields=["b_factor"])
        pLDDT = float(np.asarray(struct.b_factor, dtype=float).mean())
        return pLDDT

    def fold_chain(sequence, pdb_id, *, model, tag, out_dir=OUT_DIR):
        df = model.ttt(sequence)
        pd.DataFrame([df]).to_csv(LOGS_DIR / f"{pdb_id}_log.tsv", sep="\t", index=False)
        
        pLDDT_after = predict_structure(model, sequence, pdb_id, tag='_ttt', out_dir=out_dir)
        model.ttt_reset()
        return pLDDT_after
    
    def calculate_metrics(true_path, pred_path):
        true_struct = bsio.load_structure(true_path, extra_fields=["b_factor"])
        plddt_alpha = float(np.asarray(true_struct.b_factor, dtype=float).mean())

        tm_score = calculate_tm_score(pred_path, true_path)
        lddt = lddt_score(true_path, pred_path)

        return tm_score, lddt, plddt_alpha

    # --- Main Processing Loop ---
    start_time = time.time()
    col = 'sequence'
    len_col = 'lenghth'
    processed_count = 0
    
    print(f"{SUMMARY_PATH}")

    for i, row in df.iterrows():
        if not (start <= i < end):
            continue

        # if row[len_col] > 400 or row[len_col] < 30:
        #     continue

        if pd.isna(row[col]):
            continue
        
        start_seq_time = time.time()
        seq_id = str(row.get("id"))
        seq = str(row[col]).strip().upper()
        processed_count += 1
        
        pLDDT_before, pLDDT_after, tm_score_before, lddt_before, pldd_alphafold, tm_score_after, lddt_after = df.loc[i, [
            'pLDDT_before', 'pLDDT_after', 'tm_score_before', 'lddt_before', 'plddt_AlphaFold', 'tm_score_after', 'lddt_after']].values

        if not CALCULATE_ONLY_TTT:
    # HANDLE ALREADY PROCESSED BEFORE ------------------------------
            if (OUT_DIR / f"{seq_id}.pdb").exists():
                try:
                    tm_score_before, lddt_before, pldd_alphafold = calculate_metrics(
                        true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                        pred_path=OUT_DIR / f"{seq_id}.pdb"
                    )
                except Exception as e:
                    warnings.warn(f"Error calculating metrics in already processed: {seq_id}: {e}")
                    traceback.print_exc()

                pLDDT_before = float(np.asarray(bsio.load_structure(OUT_DIR / f"{seq_id}.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())
                
            else:
                # BEFORE ------------------------------
                try:
                    pLDDT_before = predict_structure(model, seq, seq_id, tag="")
                except Exception as e:
                    warnings.warn(f"Error calculating metrics: {seq_id}: {e}")
                    traceback.print_exc()

                try:
                    tm_score_before, lddt_before, pldd_alphafold = calculate_metrics(
                        true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                        pred_path=OUT_DIR / f"{seq_id}.pdb"
                    )
                except Exception as e:
                    warnings.warn(f"Error calculating metrics: {seq_id}: {e}")
                    traceback.print_exc()                

            df.at[i, 'pLDDT_before'] = pLDDT_before
            df.at[i, 'tm_score_before'] = tm_score_before
            df.at[i, 'lddt_before'] = lddt_before
            df.at[i, 'plddt_AlphaFold'] = pldd_alphafold

# HANDLE ALREADY PROCESSED AFTER ------------------------------ 
        if (OUT_DIR / f"{seq_id}_ttt.pdb").exists():
            try:
                tm_score_after, lddt_after, pldd_alphafold = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUT_DIR / f"{seq_id}_ttt.pdb"
                )
            except Exception as e:
                warnings.warn(f"Existed seq after metrics error: {seq_id}: {e}")
                traceback.print_exc()
            
            struct = bsio.load_structure(OUT_DIR / f"{seq_id}_ttt.pdb", extra_fields=["b_factor"])
            pLDDT_after = float(np.asarray(struct.b_factor, dtype=float).mean())
        else: 
            # AFTER ------------------------------
            try:
                pLDDT_after = fold_chain(seq, seq_id, model=model, tag="")
            except Exception as e:
                warnings.warn(f"Error folding chain, after, new {pLDDT_after}, {seq_id}: {e}")
                traceback.print_exc()

            try:
                tm_score_after, lddt_after, _ = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUT_DIR / f"{seq_id}_ttt.pdb"
                )
            except Exception as e:
                warnings.warn(f"Metrics for {seq_id}: {e}")
                traceback.print_exc()

        df.at[i, 'pLDDT_after'] = pLDDT_after
        df.at[i, 'lddt_after'] = lddt_after
        df.at[i, 'tm_score_after'] = tm_score_after
        df.at[i, 'time'] = time.time() - start_seq_time
        
        
        print(f"Processed sequence {i} (ID: {seq_id}). pLDDT before: {pLDDT_before:.2f}, after: {pLDDT_after:.2f}")

        if processed_count > 0 and processed_count % 50 == 0:
            df.to_csv(path_or_buf=SAVE_PATH, sep="\t", index=False)
            print(f"Saved progress. Total time elapsed: {time.time() - start_time:.2f} seconds for {processed_count} sequences.")

    # --- Final Save ---
    end_time = time.time()
    print(f"Processing complete. Total time elapsed: {end_time - start_time:.2f} seconds for {processed_count} sequences.")
    df.to_csv(SAVE_PATH, sep="\t", index=False)
    print("Final results saved.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run ESMFold and ProteinTTT on a chunk of sequences.")
    parser.add_argument('--chunk_start', type=int, required=True, help='Starting index of the sequence chunk.')
    parser.add_argument('--chunk_end', type=int, required=True, help='Ending index of the sequence chunk.')
    parser.add_argument('--calculate_only_ttt', type=int, default=0, help='If 1, only calculate TTT without initial ESMFold prediction.')

    args = parser.parse_args()

    if args.chunk_start >= args.chunk_end:
        print("Error: --chunk_start must be less than --chunk_end.")
        sys.exit(1)

    main(args.chunk_start, args.chunk_end, date=time.strftime("%Y%m%d"), calculate_only_ttt=args.calculate_only_ttt==1)
