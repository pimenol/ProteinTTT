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


def main(lr, ags, grad_clip_max_norm, lora_rank, lora_alpha):

    base_path = Path("/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT/data/bfvd/")
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))

    experiment_pattern = f'experement_{lr}_{ags}_{grad_clip_max_norm}_{lora_rank}_{lora_alpha}_*'
    matching_dirs = list((base_path / 'experements_msa').glob(experiment_pattern))
    if matching_dirs:
        print(f"Experiment {lr}_{ags}_{grad_clip_max_norm}_{lora_rank}_{lora_alpha} already exists")
        OUTPUTS_PATH = matching_dirs[0]
    else:
        OUTPUTS_PATH = base_path / 'experements_msa' /f'experement_{lr}_{ags}_{grad_clip_max_norm}_{lora_rank}_{lora_alpha}_{JOB_SUFFIX}'
        OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)


    OUT_DIR = OUTPUTS_PATH / 'predicted_structures'
    LOGS_DIR = OUTPUTS_PATH / 'logs' 
    SAVE_PATH = OUTPUTS_PATH / "results.tsv"
    PLOT_PATH = OUTPUTS_PATH / 'plots'

    SUMMARY_PATH = base_path / 'proteinttt_msa_testset.tsv'
    CORRECT_PREDICTED_PDB = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd")
    MSA_PATH = Path("/scratch/project/open-35-8/antonb/bfvd/bfvd_msa")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

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
    ttt_cfg.gradient_clip = True
    ttt_cfg.gradient_clip_max_norm = grad_clip_max_norm
    ttt_cfg.lora_rank = lora_rank
    ttt_cfg.lora_alpha = lora_alpha

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
        # Use buffered write for better I/O performance
        with open(out_path, 'w', buffering=8192) as f:
            f.write(pdb_str)

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
            sys.exit(1)

    def calculate_metrics(true_path, pred_path):
        tm_score = calculate_tm_score(pred_path, true_path)
        lddt = lddt_score(true_path, pred_path)

        return tm_score, lddt

    # --- Main Processing Loop ---
    col = 'sequence'
    processed_count = 0

    start_total_time = time.time()

    print(f"{SUMMARY_PATH}")
    print(f" Learning rate: {lr}, AGS: {ags}, Grad clip max norm: {grad_clip_max_norm}, LoRA rank: {model.ttt_cfg.lora_rank}, LoRA alpha: {model.ttt_cfg.lora_alpha}")

    columns_to_add = [f'pLDDT_after', f'lddt_after', f'tm_score_after']
    for col_name in columns_to_add:
        if col_name not in df.columns:
            df[col_name] = np.nan

    for i, row in df.iterrows():
        start_time = time.time()
        seq_id = str(row.get("id"))
        seq = str(row[col]).strip().upper()
        processed_count += 1

        pLDDT_after, tm_score_after, lddt_after = None, None, None

        if os.path.exists(LOGS_DIR / f'{seq_id}_log.tsv'):
            print(f"Sequence {seq_id} already processed")
            tm_score_after, lddt_after = calculate_metrics(
                    true_path=CORRECT_PREDICTED_PDB / f"{seq_id}.pdb",
                    pred_path=OUT_DIR / f"{seq_id}_ttt.pdb"
                )
            pLDDT_after = float(np.asarray(bsio.load_structure(OUT_DIR / f"{seq_id}_ttt.pdb", extra_fields=["b_factor"]).b_factor, dtype=float).mean())
        else:
            try:
                _ , pLDDT_after = fold_chain(seq, seq_id, model=model)
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

        df.at[i, 'pLDDT_after'] = pLDDT_after
        df.at[i, 'lddt_after'] = lddt_after
        df.at[i, 'tm_score_after'] = tm_score_after

        print(f"Processed sequence {i} (ID: {seq_id}), before: {df.at[i, 'pLDDT_before']:.2f}, after with MSA: {pLDDT_after}, time: {time.time() - start_time:.2f}")

    df.to_csv(SAVE_PATH, sep="\t", index=False)
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_best_plddt.png", metric='plddt')
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"lddt_vs_step_best_plddt.png", metric='lddt')
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"tm_score_vs_step_best_plddt.png", metric='tm_score')

    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_no_best_plddt.png", metric='plddt', choose_best_plddt=False)
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"lddt_vs_step_no_best_plddt.png", metric='lddt', choose_best_plddt=False)
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"tm_score_vs_step_no_best_plddt.png", metric='tm_score', choose_best_plddt=False)

    print(f"Final results saved. Total time elapsed: {time.time() - start_total_time:.2f} seconds for {processed_count} sequences.")
    warnings.warn(f"Final results saved. Total time elapsed: {time.time() - start_total_time:.2f} seconds for {processed_count} sequences.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run ESMFold and ProteinTTT on a chunk of sequences.")
    parser.add_argument('--lr', type=float, required=True, help='LR parameter for ProteinTTT.')
    parser.add_argument('--ags', type=int, required=True, help='AGS parameter for ProteinTTT.')
    parser.add_argument('--grad_clip_max_norm', type=float, required=True, help='Grad clip max norm for ProteinTTT.')
    parser.add_argument('--lora_rank', type=int, required=True, help='LoRA rank for ProteinTTT.')
    parser.add_argument('--lora_alpha', type=float, required=True, help='LoRA alpha for ProteinTTT.')

    args = parser.parse_args()

    main(args.lr, args.ags, args.grad_clip_max_norm, args.lora_rank, args.lora_alpha)
