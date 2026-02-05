import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import time
import esm
import biotite.structure.io as bsio
from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG, GRAD_CLIP_ESMFOLD_TTT_CFG
from proteinttt.utils.structure import calculate_tm_score, lddt_score
import torch
import argparse
import os
import uuid
import traceback
from proteinttt.utils.plots import plot_mean_scores_vs_step
from proteinttt.utils.fix_pdb import fix_pdb
import logging
import yaml


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


def main(config):
    """Main execution function using configuration dictionary."""
    
    # Setup paths
    base_path = Path(config['df_path'])
    if not base_path.exists():
        raise ValueError(f"df_path does not exist: {base_path}")
    
    ESM_TTT_DIR = base_path / config['output']['esm_ttt_dir']
    ESM_DIR = base_path / config['output']['esm_dir']
    LOGS_DIR = base_path / config['output']['logs_dir']
    PLOT_PATH = base_path / config['output']['plots_dir']
    SAVE_PATH = base_path / config['output']['results_file']
    SUMMARY_PATH = base_path / config['input']['summary_file']
    CORRECT_PREDICTED_PDB = base_path / config['input']['pdb_dir']
    MSA_PATH = base_path / config['input']['msa_dir']

    # Create output directories
    ESM_TTT_DIR.mkdir(parents=True, exist_ok=True)
    ESM_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(base_path / config['output']['execution_log']),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Prevent double logging from the proteinttt library
    library_logger = logging.getLogger("ttt_log")
    library_logger.propagate = False
    
    JOB_SUFFIX = os.getenv("SLURM_JOB_ID", str(uuid.uuid4()))
    logging.info(f"Job ID: {JOB_SUFFIX}")
    logging.info("="*60)
    logging.info("Configuration:")
    for key, value in config.items():
        logging.info(f"  - {key}: {value}")
    logging.info("="*60)   

    # Load data
    df = pd.read_csv(SUMMARY_PATH)
    df['sequence_length'] = df['sequence'].apply(len)
    max_len = config['max_sequence_length']
    df = df.query(f"sequence_length <= {max_len}").copy()
    logging.info(f"Loaded {len(df)} sequences (max length: {max_len})")

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = esm.pretrained.esmfold_v0().eval().to(device)
    
    # Build TTT configuration
    if config['use_gradient_clip']:
        ttt_cfg = GRAD_CLIP_ESMFOLD_TTT_CFG
    else:
        ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    
    # Apply config settings
    for key, value in config.items():
        if key not in ['df_path', 'output', 'input']:
            setattr(ttt_cfg, key, value)
    
    # Set initial chunk size
    if config.get('use_msa', False):
        base_model.set_chunk_size(128)
    
    logging.info(f"TTT config: {ttt_cfg}")

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
        logging.info(f"Processing {pdb_id} (length: {len(sequence)}, chunk_size: {chunk_size})")
        model.ttt_reset()
        try:
            if config['use_msa']:
                msa_file = MSA_PATH / f"{pdb_id}_{chain_id}.a3m"
                if not msa_file.exists():
                    logging.warning(f"MSA file not found: {msa_file}, running without MSA")
                    df = model.ttt(sequence, return_logs=True)
                else:
                    df = model.ttt(sequence, msa_pth=msa_file, return_logs=True)
            else:
                df = model.ttt(sequence, return_logs=True)
            
            pLDDT_before = save_log(df, pdb_id, chain_id)
            pLDDT_after = predict_structure(model, sequence, pdb_id, chain_id, out_dir=ESM_TTT_DIR)
            return pLDDT_before, pLDDT_after
            
        except Exception as e:
            # Reset model on any error to ensure clean state for next sequence
            logging.error(f"Error in fold_chain for {pdb_id}, resetting model: {e}")
            traceback.print_exc()
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
        seq_id = str(row.get("id"))
        chain_id = str(row.get("chain_id", 'A'))

        if config['use_true_pdb']:
            true_path = CORRECT_PREDICTED_PDB / f"{seq_id}_{chain_id}.pdb"
        else:
            true_path = None

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
            if config['use_true_pdb']:
                tm_score_ProteinTTT, lddt_ProteinTTT = calculate_metrics(
                    true_path=true_path,
                    pred_path=ESM_TTT_DIR / f"{seq_id}_{chain_id}.pdb",
                    chain_id=chain_id,
                    path_to_fix_pdb=base_path / 'predicted_structures' / 'fixed_pdb_TTT' / f"{seq_id}_{chain_id}.pdb"
                )
            else:
                tm_score_ProteinTTT, lddt_ProteinTTT = None, None
        except Exception as e:
            logging.warning(f"Metrics for {seq_id}: {e}")
            traceback.print_exc()

        df.at[i, 'pLDDT_ProteinTTT'] = pLDDT_ProteinTTT
        df.at[i, 'lddt_ProteinTTT'] = lddt_ProteinTTT
        df.at[i, 'tm_score_ProteinTTT'] = tm_score_ProteinTTT

        try:
            if config['use_true_pdb']:
                tm_score_ESMFold, lddt_ESMFold = calculate_metrics(
                    true_path=true_path,
                    pred_path=ESM_DIR / f"{seq_id}_{chain_id}.pdb",
                    chain_id=chain_id,
                    path_to_fix_pdb=base_path / 'predicted_structures' / 'fixed_pdb_ESMFold' / f"{seq_id}_{chain_id}.pdb"
                )
            else:
                tm_score_ESMFold, lddt_ESMFold = None, None
        except Exception as e:
            logging.warning(f"Metrics for {seq_id}: {e}")
            traceback.print_exc()

        df.at[i, 'pLDDT_ESMFold'] = pLDDT_ESMFold
        df.at[i, 'lddt_ESMFold'] = lddt_ESMFold
        df.at[i, 'tm_score_ESMFold'] = tm_score_ESMFold

        # Format pLDDT values safely, handling None
        pLDDT_ESMFold_str = f"{pLDDT_ESMFold:.2f}" if pLDDT_ESMFold is not None else "N/A"
        pLDDT_ProteinTTT_str = f"{pLDDT_ProteinTTT:.2f}" if pLDDT_ProteinTTT is not None else "N/A"
        logging.info(f"Processed sequence {i} (ID: {seq_id}), before: {pLDDT_ESMFold_str}, after: {pLDDT_ProteinTTT_str}, time: {time.time() - start_time:.2f}s")

    df.to_csv(SAVE_PATH, sep="\t", index=False)
    logging.info(f"Results saved to {SAVE_PATH}")
    
    # Generate plots
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_best_plddt.png", metric='plddt')
    plot_mean_scores_vs_step(LOGS_DIR, output_path=PLOT_PATH / f"plddt_vs_step_no_best_plddt.png", metric='plddt', choose_best_plddt=False)
    logging.info(f"Plots saved to {PLOT_PATH}")

    total_time = time.time() - start_total_time
    logging.info(f"="*60)
    logging.info(f"Experiment completed successfully!")
    logging.info(f"Total time elapsed: {total_time:.2f} seconds for {processed_count} sequences")
    logging.info(f"Average time per sequence: {total_time/processed_count:.2f} seconds")
    logging.info(f"="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ESMFold and ProteinTTT on sequences using a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file
  python run_df.py --config config.yaml
  
  # Override data path
  python run_df.py --config config.yaml --df_path /path/to/data
  
  # Override specific settings
  python run_df.py --config config.yaml --use_msa --steps 20
        """
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--df_path', 
        type=str,
        help='Path to the data directory (overrides config file)'
    )
    parser.add_argument(
        '--use_msa',
        action='store_true',
        help='Enable MSA mode (overrides config)'
    )
    parser.add_argument(
        '--use_gradient_clip',
        action='store_true',
        help='Enable gradient clipping (overrides config)'
    )
    parser.add_argument(
        '--use_true_pdb',
        action='store_true',
        help='Enable validation against ground truth PDB files (overrides config)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        help='Maximum sequence length to process (overrides config)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        help='Number of TTT steps (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        print("Please copy and edit config.yaml:")
        print("  cp config.yaml my_config.yaml")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from: {config_file}")
    
    # Override with command-line arguments
    if args.df_path:
        config['df_path'] = args.df_path
    
    if not config['df_path']:
        print("Error: df_path must be specified either in config file or via --df_path")
        sys.exit(1)
    
    if args.use_msa:
        config['use_msa'] = True
        config['use_gradient_clip'] = True  # MSA typically needs gradient clipping
        
    if args.use_gradient_clip:
        config['use_gradient_clip'] = True
        
    if args.use_true_pdb:
        config['use_true_pdb'] = True
        
    if args.max_length:
        config['max_sequence_length'] = args.max_length
    
    if args.steps:
        config['steps'] = args.steps
    
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Run experiment
    main(config)
