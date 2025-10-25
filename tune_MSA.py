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


def main(lr, ags):
    
    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    OUT_DIR = base_path / 'predicted_structures_msa_{lr}_{ags}'
    SUMMARY_PATH = base_path / 'proteinttt_msa_testset.tsv'
    LOGS_DIR = base_path / 'logs_msa'
    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
    MSA_PATH = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd_msa")
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))
    SAVE_PATH = base_path / f"results_{JOB_SUFFIX}.tsv"
    
    df = pd.read_csv(SUMMARY_PATH, sep="\t")

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = esm.pretrained.esmfold_v1().eval().to(device)
    ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    
    ttt_cfg.steps = 20
    ttt_cfg.seed = 0
    ttt_cfg.lr = lr
    ttt_cfg.ags = ags
    ttt_cfg.msa = True
    
    # ttt_cfg.loss_kind == "msa_soft_labels"
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

    def fold_chain(sequence, pdb_id, *, model, out_dir=OUT_DIR):
        model.ttt_reset()
        df = model.ttt(sequence, msa_pth=MSA_PATH / f"{pdb_id}.a3m", return_logs=True)
        
        df_logs = df['df'].copy()
        step_data = df['ttt_step_data']
        pdb_strings_map = {}
        for step, data_for_step in step_data.items():
            pdb_strings_map[step] = data_for_step['eval_step_preds']['pdb']
            
        df_logs['pdb'] = df_logs['step'].map(pdb_strings_map)
        
        desired_columns = ['step', 'accumulated_step', 'loss', 'eval_step_time', 'plddt', 'pdb']
        existing_columns = [col for col in desired_columns if col in df_logs.columns]
        df_formatted = df_logs[existing_columns]
        df_formatted.to_csv(Path(LOGS_DIR / f"{pdb_id}_log.tsv"), sep='\t', index=False)

        # df['df']['plddt'].iloc[-1]

        pLDDT_after = predict_structure(model, sequence, pdb_id, tag='_ttt', out_dir=out_dir)
        return pLDDT_after

    def calculate_metrics(true_path, pred_path):
        tm_score = calculate_tm_score(pred_path, true_path)
        lddt = lddt_score(true_path, pred_path)

        return tm_score, lddt

    # --- Main Processing Loop ---
    col = 'sequence'
    processed_count = 0

    print(f"{SUMMARY_PATH}")

    columns_to_add = ['pLDDT_{lr}_{ags}', 'lddt_{lr}_{ags}', 'tm_score_{lr}_{ags}']
    for col_name in columns_to_add:
        if col_name not in df.columns:
            df[col_name] = np.nan
            
    for i, row in df.iterrows():
        seq_id = str(row.get("id"))
        seq = str(row[col]).strip().upper()
        processed_count += 1

        pLDDT_before, pLDDT_after, tm_score_before, lddt_before, pldd_alphafold, tm_score_after, lddt_after = None, None, None, None, None, None, None
        
        try:
            pLDDT_after = fold_chain(seq, seq_id, model=model)
        except Exception as e:
            warnings.warn(f"Error folding chain {pLDDT_after}, {seq_id}: {e}")
            traceback.print_exc()

        try:
            tm_score_after, lddt_after = calculate_metrics(
                true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                pred_path=OUT_DIR / f"{seq_id}_ttt.pdb"
            )
        except Exception as e:
            warnings.warn(f"Metrics for {seq_id}: {e}")
            traceback.print_exc()

        df.at[i, columns_to_add[0]] = pLDDT_after
        df.at[i, columns_to_add[1]] = lddt_after
        df.at[i, columns_to_add[2]] = tm_score_after

        print(f"Processed sequence {i} (ID: {seq_id}). pLDDT before: {pLDDT_before:.2f}, after: {pLDDT_after:.2f}")
        
    df.to_csv(SAVE_PATH, sep="\t", index=False)
    print("Final results saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ESMFold and ProteinTTT on a chunk of sequences.")
    parser.add_argument('--lr', type=float, required=True, help='LR parameter for ProteinTTT.')
    parser.add_argument('--ags', type=int, required=True, help='AGS parameter for ProteinTTT.')

    args = parser.parse_args()

    main(args.lr, args.ags)
