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


def main(start, end, date):
    # --- Configuration ---
    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    OUTPUT_PDB = base_path / 'predicted_structures'
    SUMMARY_PATH = base_path / 'to_process.tsv'
    SAVE_PATH = base_path / f"results_after_{start}_{end}_{date}.tsv"
    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
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

    def predict_structure(model, sequence, pdb_id, tag, out_dir=OUTPUT_PDB):
        with torch.no_grad():
            pdb_str = model.infer_pdb(sequence)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        out_path = out_dir / f"{pdb_id}{tag}.pdb"
        out_path.write_text(pdb_str)

        struct = bsio.load_structure(out_path, extra_fields=["b_factor"])
        pLDDT = float(np.asarray(struct.b_factor, dtype=float).mean())
        return pLDDT

    def fold_chain(sequence, pdb_id, *, model, tag, out_dir=OUTPUT_PDB):
        model.ttt(sequence)
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

        if row[len_col] > 400 or row[len_col] < 30:
            continue

        if pd.isna(row[col]):
            continue
        
        start_seq_time = time.time()
        seq_id = str(row.get("id"))
        seq = str(row[col]).strip().upper()

        pLDDT_before, pLDDT_after, tm_score_before, lddt_before, pldd_alphafold, tm_score_after, lddt_after = df.loc[i, [
            'pLDDT_before', 'pLDDT_after', 'tm_score_before', 'lddt_before', 'plddt_AlphaFold', 'tm_score_after', 'lddt_after']].values

# HANDLE ALREADY PROCESSED BEFORE ------------------------------
        # if (OUTPUT_PDB / f"{seq_id}.pdb").exists():
        #     try:
        #         tm_score_before, lddt_before, pldd_alphafold = calculate_metrics(
        #             true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
        #             pred_path=OUTPUT_PDB / f"{seq_id}.pdb"
        #         )
        #     except Exception as e:
        #         warnings.warn(f"Error calculating metrics in already processed: {seq_id}: {e}")

        #     pLDDT_before = float(np.asarray(bsio.load_structure(OUTPUT_PDB / f"{seq_id}.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())
            
        # else:
        #     # BEFORE ------------------------------
        #     try:
        #         pLDDT_before = predict_structure(model, seq, seq_id, tag="")
        #     except Exception as e:
        #         continue

        #     try:
        #         tm_score_before, lddt_before, pldd_alphafold = calculate_metrics(
        #             true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
        #             pred_path=OUTPUT_PDB / f"{seq_id}.pdb"
        #         )
        #     except Exception as e:
        #         warnings.warn(f"Error calculating metrics: {seq_id}: {e}")

        # df.at[i, 'pLDDT_before'] = pLDDT_before
        # df.at[i, 'tm_score_before'] = tm_score_before
        # df.at[i, 'lddt_before'] = lddt_before
        # df.at[i, 'plddt_AlphaFold'] = pldd_alphafold

# HANDLE ALREADY PROCESSED AFTER ------------------------------ 
        if (OUTPUT_PDB / f"{seq_id}_ttt.pdb").exists():
            try:
                tm_score_after, lddt_after, pldd_alphafold = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUTPUT_PDB / f"{seq_id}_ttt.pdb"
                )
            except Exception as e:
                warnings.warn(f"Existed seq after metrics error: {seq_id}: {e}")
            
            struct = bsio.load_structure(OUTPUT_PDB / f"{seq_id}_ttt.pdb", extra_fields=["b_factor"])
            pLDDT_after = float(np.asarray(struct.b_factor, dtype=float).mean())
        else: 
            # AFTER ------------------------------
            try:
                pLDDT_after = fold_chain(seq, seq_id, model=model, tag="")
            except Exception as e:
                warnings.warn(f"Error folding chain, after, new {pLDDT_after}, {seq_id}: {e}")

            try:
                tm_score_after, lddt_after, _ = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUTPUT_PDB / f"{seq_id}_ttt.pdb"
                )
            except Exception as e:
                warnings.warn(f"Metrics for {seq_id}: {e}")

        df.at[i, 'pLDDT_after'] = pLDDT_after
        df.at[i, 'lddt_after'] = lddt_after
        df.at[i, 'tm_score_after'] = tm_score_after
        df.at[i, 'time'] = time.time() - start_seq_time
        processed_count += 1
        
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

    args = parser.parse_args()

    if args.chunk_start >= args.chunk_end:
        print("Error: --chunk_start must be less than --chunk_end.")
        sys.exit(1)

    main(args.chunk_start, args.chunk_end, date=time.strftime("%Y%m%d"))
    
    # if len(sys.argv) != 3:
    #     print("Usage: python your_script_name.py <start_index> <end_index>")
    #     sys.exit(1)

    # try:
    #     start_index = int(sys.argv[1])
    #     end_index = int(sys.argv[2])
    #     main(start_index, end_index, date=time.strftime("%Y%m%d"))
    # except ValueError:
    #     print("Error: Start and end indices must be integers.")
    #     sys.exit(1)
