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
from proteinttt.utils.plots import plot_mean_scores_vs_step
from proteinttt.utils.fix_pdb import fix_pdb


# def check_model_weights(model):
#     """Check if model has NaN or Inf values in weights."""
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             if torch.isnan(param).any():
#                 return False, f"NaN detected in {name}"
#             if torch.isinf(param).any():
#                 return False, f"Inf detected in {name}"
#     return True, "Model weights are valid"


def set_dynamic_chunk_size(model, sequence_length):
    """Dynamically set chunk size based on sequence length."""
    if sequence_length < 100:
        chunk_size = 256
    elif sequence_length < 200:
        chunk_size = 128
    elif sequence_length < 400:
        chunk_size = 64
    elif sequence_length < 500:
        chunk_size = 32
    elif sequence_length < 600:
        chunk_size = 16
    elif sequence_length < 700:
        chunk_size = 8
    else:
        chunk_size = 4
    
    model.set_chunk_size(chunk_size)
    return chunk_size


def main():

    USE_MSA = False

    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/cameo")
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))

    OUTPUTS_PATH =  (base_path / 'cameo_msa' if USE_MSA else base_path)

    ESM_TTT_DIR = OUTPUTS_PATH / 'predicted_structures' / 'ESMFold_ProteinTTT'
    ESM_DIR = OUTPUTS_PATH / 'predicted_structures' / 'ESMFold'
    LOGS_DIR = OUTPUTS_PATH / 'logs' 
    SAVE_PATH = OUTPUTS_PATH / "results.tsv"
    PLOT_PATH = OUTPUTS_PATH / 'plots'

    SUMMARY_PATH = base_path / 'summary.csv'
    CORRECT_PREDICTED_PDB = base_path / "pdb"
    MSA_PATH = base_path / 'msa'

    ESM_TTT_DIR.mkdir(parents=True, exist_ok=True)
    ESM_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_PATH)
    df = df.query("sequence_length <= 500").copy()

    print(f"Configuration: USE_MSA={USE_MSA}")
    print(f"Output path: {OUTPUTS_PATH}")

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = esm.pretrained.esmfold_v0().eval().to(device)
    
    ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    ttt_cfg.msa = USE_MSA
    if USE_MSA:
        base_model.set_chunk_size(128)
        ttt_cfg.gradient_clip = True
        ttt_cfg.gradient_clip_max_norm = 1.0
        ttt_cfg.lora_rank = 64
        ttt_cfg.lora_alpha = 128.0
        ttt_cfg.steps = 30
        ttt_cfg.seed = 0
        ttt_cfg.lr = 0.04
        ttt_cfg.ags = 32
    # learning rate scheduler
    # ttt_cfg.lr_scheduler = "cosine_warmup"
    # ttt_cfg.lr_warmup_steps = 5  # optimizer steps (not micro steps)
    # ttt_cfg.lr_min = 0.0

    # ttt_cfg.loss_kind == "msa_soft_labels"
    model = ESMFoldTTT.ttt_from_pretrained(
        base_model,
        ttt_cfg=ttt_cfg,
        esmfold_config=base_model.cfg
    ).to(device)

    def predict_structure(model, sequence, pdb_id, chain_id, out_dir=ESM_TTT_DIR):
        with torch.no_grad():
            pdb_str = model.infer_pdb(sequence)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        out_path = out_dir / f"{pdb_id}_{chain_id}.pdb"
        with open(out_path, 'w', buffering=8192) as f:
            f.write(pdb_str)

        struct = bsio.load_structure(out_path, extra_fields=["b_factor"])
        pLDDT = float(np.asarray(struct.b_factor, dtype=float).mean())
        return pLDDT

    def save_log(df, pdb_id, chain_id):
        df_logs = df['df'].copy()
        step_data = df['ttt_step_data']

        pdb_strings_map = {}
        for step, data_for_step in step_data.items():
            pdb_strings_map[step] = data_for_step['eval_step_preds']['pdb'][0]

        df_logs['pdb'] = df_logs['step'].map(pdb_strings_map)

        desired_columns = ['step', 'accumulated_step', 'loss', 'score_seq_time', 'eval_step_time', 'plddt', 'pdb']
        existing_columns = [col for col in desired_columns if col in df_logs.columns]
        df_combined_logs = df_logs[existing_columns]
        df_combined_logs.to_csv(Path(LOGS_DIR / f"{pdb_id}_{chain_id}_log.tsv"), sep='\t', index=False)

        pdb_before = df_combined_logs['pdb'].iloc[0]
        out_path = ESM_DIR / f"{pdb_id}_{chain_id}.pdb"
        with open(out_path, 'w', buffering=8192) as f:
            f.write(pdb_before)

        pLDDT_before = df_combined_logs['plddt'].iloc[0]
        return pLDDT_before

    def fold_chain(sequence, pdb_id, chain_id, model):
        chunk_size = set_dynamic_chunk_size(model, len(sequence))
        print(f"Processing {pdb_id} (length: {len(sequence)}, chunk_size: {chunk_size})")
        model.ttt_reset()
        try:
            if USE_MSA:
                df = model.ttt(sequence, msa_pth=MSA_PATH / f"{pdb_id}_{chain_id}.a3m", return_logs=True)
            else:
                df = model.ttt(sequence, return_logs=True)
            
            pLDDT_before = save_log(df, pdb_id, chain_id)
            pLDDT_after = predict_structure(model, sequence, pdb_id, chain_id, out_dir=ESM_TTT_DIR)
            return pLDDT_before, pLDDT_after
            
        except Exception as e:
            # Reset model on any error to ensure clean state for next sequence
            warnings.warn(f"Error in fold_chain for {pdb_id}, resetting model: {e}")
            sys.exit(1)

    def calculate_metrics(true_path, pred_path, chain_id, path_to_fix_pdb):
        fix_pdb(true_path, pred_path, chain_id, path_to_fix_pdb)

        tm_score = calculate_tm_score(path_to_fix_pdb, true_path)
        lddt = lddt_score(true_path, path_to_fix_pdb)

        return tm_score, lddt

    # --- Main Processing Loop ---
    col = 'sequence'
    processed_count = 0

    start_total_time = time.time()

    print(f"{SUMMARY_PATH}")

    columns_to_add = [f'pLDDT_ProteinTTT', f'lddt_ProteinTTT', f'tm_score_ProteinTTT', 'pLDDT_ESMFold', 'lddt_ESMFold', 'tm_score_ESMFold']
    for col_name in columns_to_add:
        if col_name not in df.columns:
            df[col_name] = np.nan

    for i, row in df.iterrows():
        start_time = time.time()
        seq_id = str(row.get("pdb_id"))
        chain_id = str(row.get("chain_id"))

        true_path = CORRECT_PREDICTED_PDB / f"{seq_id}_{chain_id}.pdb"

        seq = str(row[col]).strip().upper()
        processed_count += 1

        pLDDT_ProteinTTT, tm_score_ProteinTTT, lddt_ProteinTTT = None, None, None
        pLDDT_ESMFold, tm_score_ESMFold, lddt_ESMFold = None, None, None

        if not (ESM_TTT_DIR / f"{seq_id}_{chain_id}.pdb").exists():
            try:
                pLDDT_ESMFold, pLDDT_ProteinTTT = fold_chain(seq, seq_id, chain_id, model=model)
            except Exception as e:
                df.to_csv(SAVE_PATH, sep="\t", index=False)
                warnings.warn(f"Error folding chain {seq_id}: {e}")
                traceback.print_exc()
        else:
            pLDDT_ProteinTTT = float(np.asarray(bsio.load_structure(ESM_TTT_DIR / f"{seq_id}_{chain_id}.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())
            pLDDT_ESMFold = float(np.asarray(bsio.load_structure(ESM_DIR / f"{seq_id}_{chain_id}.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())
        try:
            if pLDDT_ProteinTTT is not None:  # Only calculate metrics if folding succeeded
                tm_score_ProteinTTT, lddt_ProteinTTT = calculate_metrics(
                    true_path=true_path,
                    pred_path=ESM_TTT_DIR / f"{seq_id}_{chain_id}.pdb",
                    chain_id=chain_id,
                    path_to_fix_pdb= OUTPUTS_PATH / 'predicted_structures' / 'fixed_pdb_TTT' / f"{seq_id}_{chain_id}.pdb"
                )
        except Exception as e:
            warnings.warn(f"Metrics for {seq_id}: {e}")
            traceback.print_exc()

        df.at[i, 'pLDDT_ProteinTTT'] = pLDDT_ProteinTTT
        df.at[i, 'lddt_ProteinTTT'] = lddt_ProteinTTT
        df.at[i, 'tm_score_ProteinTTT'] = tm_score_ProteinTTT

        try:
            tm_score_ESMFold, lddt_ESMFold = calculate_metrics(
                true_path=true_path,
                pred_path=ESM_DIR / f"{seq_id}_{chain_id}.pdb",
                chain_id=chain_id,
                path_to_fix_pdb= OUTPUTS_PATH / 'predicted_structures' / 'fixed_pdb_ESMFold' / f"{seq_id}_{chain_id}.pdb"
            )
        except Exception as e:
            warnings.warn(f"Metrics for {seq_id}: {e}")
            traceback.print_exc()

        df.at[i, 'pLDDT_ESMFold'] = pLDDT_ESMFold
        df.at[i, 'lddt_ESMFold'] = lddt_ESMFold
        df.at[i, 'tm_score_ESMFold'] = tm_score_ESMFold

        # Format pLDDT values safely, handling None
        pLDDT_ESMFold_str = f"{pLDDT_ESMFold:.2f}" if pLDDT_ESMFold is not None else "N/A"
        pLDDT_ProteinTTT_str = f"{pLDDT_ProteinTTT:.2f}" if pLDDT_ProteinTTT is not None else "N/A"
        print(f"Processed sequence {i} (ID: {seq_id}), before: {pLDDT_ESMFold_str}, after: {pLDDT_ProteinTTT_str}, time: {time.time() - start_time:.2f}")

    df.to_csv(SAVE_PATH, sep="\t", index=False)
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_best_plddt.png", metric='plddt')
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_no_best_plddt.png", metric='plddt', choose_best_plddt=False)

    print(f"Final results saved. Total time elapsed: {time.time() - start_total_time:.2f} seconds for {processed_count} sequences.")
    warnings.warn(f"Final results saved. Total time elapsed: {time.time() - start_total_time:.2f} seconds for {processed_count} sequences.")


if __name__ == "__main__":
    
    main()
