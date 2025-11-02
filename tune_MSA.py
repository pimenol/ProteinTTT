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


def check_model_weights(model):
    """Check if model has NaN or Inf values in weights."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                return False, f"NaN detected in {name}"
            if torch.isinf(param).any():
                return False, f"Inf detected in {name}"
    return True, "Model weights are valid"


def set_dynamic_chunk_size(model, sequence_length):
    """Dynamically set chunk size based on sequence length."""
    if sequence_length < 100:
        chunk_size = 256
    elif sequence_length < 200:
        chunk_size = 128
    elif sequence_length < 400:
        chunk_size = 64
    else:
        chunk_size = 32
    
    model.set_chunk_size(chunk_size)
    return chunk_size


def main(lr, ags):

    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    OUT_DIR = base_path /'predicted_structures_msa'/ f'predicted_structures_msa_{lr}_{ags}'
    SUMMARY_PATH = base_path / 'proteinttt_msa_testset.tsv'
    LOGS_DIR = base_path / 'logs_msa' / f'logs_msa_{lr}_{ags}'
    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
    MSA_PATH = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd_msa")
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))
    SAVE_PATH = base_path / 'results_msa' / f"results_{JOB_SUFFIX}.tsv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # SAVE_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_PATH, sep="\t")

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = esm.pretrained.esmfold_v1().eval().to(device)
    base_model.set_chunk_size(128)
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
    
    # Store initial model state for potential resets
    initial_model_state = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}

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

    def save_log(df, pdb_id):
        df_logs = df['df'].copy()
        step_data = df['ttt_step_data']

        pdb_strings_map = {}
        for step, data_for_step in step_data.items():
            pdb_strings_map[step] = data_for_step['eval_step_preds']['pdb'][0]

        df_logs['pdb'] = df_logs['step'].map(pdb_strings_map)

        desired_columns = ['step', 'accumulated_step', 'loss', 'score_seq_time', 'eval_step_time', 'plddt', 'tm_score', 'lddt', 'pdb']
        existing_columns = [col for col in desired_columns if col in df_logs.columns]
        df_combined_logs = df_logs[existing_columns]
        df_combined_logs.to_csv(Path(LOGS_DIR / f"{pdb_id}_log.tsv"), sep='\t', index=False)

        pLDDT_before = df_combined_logs['plddt'].iloc[0]

        return pLDDT_before

    def fold_chain(sequence, pdb_id, *, model, out_dir=OUT_DIR):
        # Set dynamic chunk size based on sequence length
        chunk_size = set_dynamic_chunk_size(model, len(sequence))
        print(f"Processing {pdb_id} (length: {len(sequence)}, chunk_size: {chunk_size})")
        
        model.ttt_reset()
        
        # Run TTT with error checking
        try:
            df = model.ttt(sequence, msa_pth=MSA_PATH / f"{pdb_id}.a3m", return_logs=True, correct_pdb_path=CORRECT_PREDICTED_PDB / f"{pdb_id}.pdb")
            weights_valid, msg = check_model_weights(model)
            if not weights_valid:
                warnings.warn(f"Model weights became invalid during TTT for {pdb_id}: {msg}")
                # Reset model for next iteration
                for name, param in model.named_parameters():
                    if param.requires_grad and name in initial_model_state:
                        param.data.copy_(initial_model_state[name])
                model.ttt_reset()
                raise RuntimeError(f"Model weights became invalid: {msg}")
            
            pLDDT_before = save_log(df, pdb_id)
            pLDDT_after = predict_structure(model, sequence, pdb_id, tag='_ttt', out_dir=out_dir)
            return pLDDT_before, pLDDT_after
            
        except Exception as e:
            # Reset model on any error to ensure clean state for next sequence
            warnings.warn(f"Error in fold_chain for {pdb_id}, resetting model: {e}")
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_model_state:
                    param.data.copy_(initial_model_state[name])
            model.ttt_reset()
            raise

    def calculate_metrics(true_path, pred_path):
        tm_score = calculate_tm_score(pred_path, true_path)
        lddt = lddt_score(true_path, pred_path)

        return tm_score, lddt

    # --- Main Processing Loop ---
    col = 'sequence'
    processed_count = 0

    print(f"{SUMMARY_PATH}")
    print(f" Learning rate: {lr}, AGS: {ags}")

    columns_to_add = [f'pLDDT_{lr}_{ags}', f'lddt_{lr}_{ags}', f'tm_score_{lr}_{ags}']
    for col_name in columns_to_add:
        if col_name not in df.columns:
            df[col_name] = np.nan

    for i, row in df.iterrows():
        seq_id = str(row.get("id"))
        seq = str(row[col]).strip().upper()
        processed_count += 1

        pLDDT_before, pLDDT_after, tm_score_after, lddt_after = None, None, None, None

        try:
            pLDDT_before, pLDDT_after = fold_chain(seq, seq_id, model=model)
        except Exception as e:
            warnings.warn(f"Error folding chain {seq_id}: {e}")
            traceback.print_exc()
            # Ensure model is in a good state for next iteration
            try:
                weights_valid, msg = check_model_weights(model)
                if not weights_valid:
                    warnings.warn(f"Resetting model after error: {msg}")
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in initial_model_state:
                            param.data.copy_(initial_model_state[name])
                    model.ttt_reset()
            except Exception as reset_error:
                warnings.warn(f"Error resetting model: {reset_error}")

        try:
            if pLDDT_after is not None:  # Only calculate metrics if folding succeeded
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

        # Safe printing with None handling
        plddt_before_str = f"{pLDDT_before:.2f}" if pLDDT_before is not None else "N/A"
        plddt_after_str = f"{pLDDT_after:.2f}" if pLDDT_after is not None else "N/A"
        plddt_df_before = df.at[i, 'pLDDT_before'] if 'pLDDT_before' in df.columns else None
        plddt_df_before_str = f"{plddt_df_before:.2f}" if plddt_df_before is not None and not pd.isna(plddt_df_before) else "N/A"
        
        print(f"Processed sequence {i} (ID: {seq_id}). pLDDT before df: {plddt_df_before_str}, predicted: {plddt_before_str}, after: {plddt_after_str}")

    df.to_csv(SAVE_PATH, sep="\t", index=False)
    print("Final results saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ESMFold and ProteinTTT on a chunk of sequences.")
    parser.add_argument('--lr', type=float, required=True, help='LR parameter for ProteinTTT.')
    parser.add_argument('--ags', type=int, required=True, help='AGS parameter for ProteinTTT.')

    args = parser.parse_args()

    main(args.lr, args.ags)
