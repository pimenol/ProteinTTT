#!/usr/bin/env python3
"""
FGR ProteinGym Fitness Prediction Script

Analyzes how FGR (Fidelity-Gain Ratio) metrics work for fitness prediction 
using ESM2 650M + ProteinTTT on ProteinGym benchmark proteins.

For each protein:
1. Uses ESM2 650M to predict baseline fitness scores (masked-marginals)
2. Runs ProteinTTT for 30 steps
3. At each step, records FGR metrics and computes Spearman correlation with experimental fitness

Usage:
    python run_fgr_proteingym.py --config fgr_proteingym_config.yaml
    python run_fgr_proteingym.py --config fgr_proteingym_config.yaml --dms_index 0
"""

import sys
import os
import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats

import esm
from proteinttt.models.esm2 import ESM2TTT, DEFAULT_ESM2_650M_TTT_CFG
from proteinttt.base import TTTConfig


def setup_logging(log_level: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("fgr_proteingym")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def select_proteins(
    reference_df: pd.DataFrame, 
    max_length: int, 
    num_proteins: int,
    specific_dms_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """Select proteins from ProteinGym/MaveDB reference file based on criteria."""
    
    if specific_dms_ids:
        # Use specified DMS IDs
        selected = reference_df[reference_df['DMS_id'].isin(specific_dms_ids)].copy()
    else:
        # Filter by sequence length and select top N
        filtered = reference_df[reference_df['seq_len'] <= max_length].copy()
        
        # Use DMS_total_number_mutants if available (ProteinGym format)
        # Otherwise use mutations_total_parsed or DMS_total_number_mutants (MaveDB format)
        mutant_col = None
        for col in ['DMS_total_number_mutants', 'mutations_total_parsed', 'num_variants_reported']:
            if col in filtered.columns:
                mutant_col = col
                break
        
        if mutant_col:
            # Sort by number of mutants (prefer larger datasets) and sequence length
            filtered = filtered.sort_values(
                by=[mutant_col, 'seq_len'], 
                ascending=[False, True]
            )
        else:
            # Just sort by sequence length
            filtered = filtered.sort_values(by='seq_len', ascending=True)
        
        selected = filtered.head(num_proteins)
    
    return selected


def get_optimal_window(mutation_position_relative: int, seq_len_wo_special: int, model_window: int) -> Tuple[int, int]:
    """Get optimal window for scoring long sequences (from ProteinGym)."""
    half_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return 0, seq_len_wo_special
    elif mutation_position_relative < half_window:
        return 0, model_window
    elif mutation_position_relative >= seq_len_wo_special - half_window:
        return seq_len_wo_special - model_window, seq_len_wo_special
    else:
        return mutation_position_relative - half_window, mutation_position_relative + half_window


def compute_masked_marginals_scores(
    model: torch.nn.Module,
    alphabet,
    sequence: str,
    df_mutations: pd.DataFrame,
    mutant_col: str = 'mutant',
    offset_idx: int = 1,
    device: torch.device = torch.device('cuda'),
    model_window: int = 1024,
) -> np.ndarray:
    """Compute masked-marginals fitness scores for all mutations."""
    
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)
    
    # Compute all token probabilities via masked marginals
    all_token_probs = []
    seq_len = len(sequence) + 2  # +2 for BOS and EOS tokens
    
    for i in tqdm(range(seq_len), desc="Computing masked marginals", leave=False):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        
        # Handle long sequences
        if seq_len > model_window:
            start, end = get_optimal_window(i, seq_len, model_window)
            batch_tokens_masked_window = batch_tokens_masked[:, start:end]
        else:
            start = 0
            batch_tokens_masked_window = batch_tokens_masked
        
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked_window)["logits"], dim=-1
            )
        all_token_probs.append(token_probs[:, i - start].detach().cpu())
    
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    
    # Score each mutation
    scores = []
    for _, row in df_mutations.iterrows():
        mutation = row[mutant_col]
        score = 0.0
        
        # Handle multiple mutations (e.g., "A1B:C2D")
        for mut in mutation.split(":"):
            wt = mut[0]
            idx = int(mut[1:-1]) - offset_idx
            mt = mut[-1]
            
            # Verify wild-type
            if idx < len(sequence) and sequence[idx] == wt:
                wt_encoded = alphabet.get_idx(wt)
                mt_encoded = alphabet.get_idx(mt)
                # +1 for BOS token
                score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
        
        scores.append(score)
    
    return np.array(scores)


def compute_fitness_scores_at_step(
    model: torch.nn.Module,
    alphabet,
    sequence: str,
    df_mutations: pd.DataFrame,
    mutant_col: str = 'mutant',
    offset_idx: int = 1,
    device: torch.device = torch.device('cuda'),
    model_window: int = 1024,
) -> np.ndarray:
    """Compute fitness scores using current model state."""
    return compute_masked_marginals_scores(
        model=model,
        alphabet=alphabet,
        sequence=sequence,
        df_mutations=df_mutations,
        mutant_col=mutant_col,
        offset_idx=offset_idx,
        device=device,
        model_window=model_window,
    )


def compute_spearman(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Spearman correlation coefficient."""
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
    if mask.sum() < 2:
        return float('nan')
    
    corr, _ = stats.spearmanr(predictions[mask], ground_truth[mask])
    return corr


class FitnessEvaluatorESM2TTT(ESM2TTT):
    """Extended ESM2TTT that evaluates fitness during TTT."""
    
    def __init__(self, ttt_cfg: TTTConfig, df_mutations: pd.DataFrame, 
                 ground_truth: np.ndarray, sequence: str, 
                 mutant_col: str = 'mutant', offset_idx: int = 1,
                 model_window: int = 1024, **kwargs):
        super().__init__(ttt_cfg, **kwargs)
        self.df_mutations = df_mutations
        self.ground_truth = ground_truth
        self.sequence = sequence
        self.mutant_col = mutant_col
        self.offset_idx = offset_idx
        self.model_window = model_window
        self._fitness_scores_per_step = {}
        self._spearman_per_step = {}
    
    def _ttt_eval_step(
        self,
        step: int,
        loss,
        perplexity: float,
        all_log_probs,
        seq: str,
        msa_pth,
        x: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Extended eval step that computes fitness scores and Spearman correlation."""
        # Call parent eval_step for FGR metrics
        eval_step_preds, eval_step_metrics, confidence = super()._ttt_eval_step(
            step=step,
            loss=loss,
            perplexity=perplexity,
            all_log_probs=all_log_probs,
            seq=seq,
            msa_pth=msa_pth,
            x=x,
            **kwargs,
        )
        
        # Compute fitness scores at this step
        device = next(self.parameters()).device
        
        fitness_start_time = time.time()
        fitness_scores = compute_fitness_scores_at_step(
            model=self,
            alphabet=self.ttt_alphabet,
            sequence=self.sequence,
            df_mutations=self.df_mutations,
            mutant_col=self.mutant_col,
            offset_idx=self.offset_idx,
            device=device,
            model_window=self.model_window,
        )
        fitness_time = time.time() - fitness_start_time
        
        # Compute Spearman correlation
        spearman = compute_spearman(fitness_scores, self.ground_truth)
        
        # Store results
        self._fitness_scores_per_step[step] = fitness_scores
        self._spearman_per_step[step] = spearman
        
        # Add to metrics
        eval_step_metrics['spearman'] = spearman
        eval_step_metrics['fitness_eval_time'] = fitness_time
        
        return eval_step_preds, eval_step_metrics, perplexity if perplexity else confidence


def run_protein_experiment(
    dms_row: pd.Series,
    config: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """Run FGR analysis for a single protein."""
    
    dms_id = dms_row['DMS_id']
    sequence = dms_row['target_seq']
    dms_filename = dms_row['DMS_filename']
    
    # Handle different column naming conventions (ProteinGym vs MaveDB)
    mutant_col = 'mutant'  # Default
    if 'DMS_mutant_column' in dms_row and pd.notna(dms_row['DMS_mutant_column']):
        mutant_col = dms_row['DMS_mutant_column']
    
    # Handle sequence offset
    offset_idx = 1  # Default 1-based indexing
    if 'start_idx' in dms_row and pd.notna(dms_row['start_idx']):
        offset_idx = int(dms_row['start_idx'])
    elif 'sequence_offset' in dms_row and pd.notna(dms_row['sequence_offset']):
        offset_idx = int(dms_row['sequence_offset'])
    
    logger.info(f"Processing {dms_id} (length: {len(sequence)})")
    
    # Check if output already exists - sanitize filename for special characters
    safe_dms_id = dms_id.replace(':', '_').replace('/', '_')
    output_file = output_dir / f"{safe_dms_id}_ttt.csv"
    if output_file.exists():
        logger.info(f"Output already exists for {dms_id}, skipping...")
        return pd.read_csv(output_file)
    
    # Load DMS data
    data_folder = Path(config['data_folder'])
    dms_path = data_folder / config['dms_data_folder'] / dms_filename
    
    if not dms_path.exists():
        logger.warning(f"DMS file not found: {dms_path}")
        return None
    
    df_mutations = pd.read_csv(dms_path)
    
    # Find mutant column
    mutant_col_candidates = ['mutant', 'hgvs_pro', 'variant', 'mutation', 'aa_substitutions']
    for candidate in mutant_col_candidates:
        if candidate in df_mutations.columns:
            mutant_col = candidate
            break
    
    if mutant_col not in df_mutations.columns:
        logger.warning(f"Could not find mutant column in {dms_filename}. Available: {df_mutations.columns.tolist()}")
        return None
    
    # Get ground truth fitness values
    # Common column names for fitness values
    fitness_cols = ['score', 'DMS_score', 'fitness', 'effect', 'log_fitness', 'functional_score', 'mean']
    gt_col = None
    for col in fitness_cols:
        if col in df_mutations.columns:
            gt_col = col
            break
    
    if gt_col is None:
        # Try to find any numeric column that looks like fitness
        for col in df_mutations.columns:
            if col != mutant_col and df_mutations[col].dtype in [np.float64, np.float32, np.int64]:
                gt_col = col
                break
    
    if gt_col is None:
        logger.warning(f"Could not find fitness column in {dms_filename}")
        return None
    
    # Filter out rows with missing values
    df_mutations = df_mutations.dropna(subset=[mutant_col, gt_col])
    
    # Filter to single mutations only (no ":" in mutation string for multi-mutants)
    df_mutations = df_mutations[~df_mutations[mutant_col].astype(str).str.contains(':')]
    
    # Remove duplicates if any (keep first)
    df_mutations = df_mutations.drop_duplicates(subset=[mutant_col], keep='first')
    
    ground_truth = df_mutations[gt_col].values
    logger.info(f"Using '{gt_col}' as fitness column, '{mutant_col}' as mutation column, {len(df_mutations)} mutations")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load ESM2 model
    logger.info(f"Loading model: {config['model_checkpoint']}")
    base_model, alphabet = esm.pretrained.load_model_and_alphabet(config['model_checkpoint'])
    base_model = base_model.eval().to(device)
    
    # Build TTT configuration (ensure proper types)
    ttt_cfg = TTTConfig(
        lr=float(config['ttt']['lr']),
        ags=int(config['ttt']['ags']),
        steps=int(config['ttt']['steps']),
        batch_size=int(config['ttt']['batch_size']),
        loss_kind=str(config['ttt']['loss_kind']),
        score_seq_kind=str(config['ttt']['score_seq_kind']) if config['ttt']['score_seq_kind'] else None,
        score_seq_steps_list=config['ttt']['score_seq_steps_list'],
        eval_each_step=bool(config['ttt']['eval_each_step']),
        fgr_enabled=bool(config['ttt']['fgr_enabled']),
        fgr_drift_threshold=float(config['ttt']['fgr_drift_threshold']),
        fgr_ratio_threshold=float(config['ttt']['fgr_ratio_threshold']),
        fgr_early_stopping=bool(config['ttt']['fgr_early_stopping']),
        seed=int(config['ttt']['seed']),
    )
    
    # Create model with fitness evaluation capability
    model = FitnessEvaluatorESM2TTT.ttt_from_pretrained(
        base_model,
        ttt_cfg=ttt_cfg,
        df_mutations=df_mutations,
        ground_truth=ground_truth,
        sequence=sequence,
        mutant_col=mutant_col,
        offset_idx=offset_idx,
    )
    model = model.to(device)
    
    # Run TTT
    logger.info(f"Running ProteinTTT for {config['ttt']['steps']} steps...")
    start_time = time.time()
    
    try:
        ttt_result = model.ttt(sequence, return_logs=True)
    except Exception as e:
        logger.error(f"Error during TTT for {dms_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    total_time = time.time() - start_time
    logger.info(f"TTT completed in {total_time:.2f}s")
    
    # Extract results
    df_logs = ttt_result['df'].copy()
    
    # Add Spearman correlations
    df_logs['spearman'] = df_logs['step'].map(model._spearman_per_step)
    
    # Add metadata
    df_logs['dms_id'] = dms_id
    df_logs['seq_len'] = len(sequence)
    df_logs['num_mutations'] = len(df_mutations)
    df_logs['total_time'] = total_time
    
    # Rename 'spearman' to 'avg_spearman' for consistency with request
    if 'spearman' in df_logs.columns:
        df_logs = df_logs.rename(columns={'spearman': 'avg_spearman'})
    
    # Reorder columns to match requested format
    cols_order = [
        'step', 'accumulated_step', 'loss', 'perplexity',
        'ttt_step_time', 'score_seq_time', 'eval_step_time', 'lr',
        'fgr_ratio', 'fgr_loss_delta', 'fgr_drift_delta',
        'fgr_ratio_cumulative', 'fgr_drift', '_fgr_cumulative_loss_gain',
        'fgr_ema_ratio', 'avg_spearman', 'fitness_eval_time',
    ]
    # Keep only existing columns in order
    cols_order = [c for c in cols_order if c in df_logs.columns]
    remaining = [c for c in df_logs.columns if c not in cols_order]
    df_logs = df_logs[cols_order + remaining]
    
    # Save results
    df_logs.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")
    
    # Log final Spearman
    final_spearman = df_logs['avg_spearman'].iloc[-1] if 'avg_spearman' in df_logs.columns else None
    initial_spearman = df_logs['avg_spearman'].iloc[0] if 'avg_spearman' in df_logs.columns else None
    if initial_spearman is not None and final_spearman is not None:
        logger.info(f"{dms_id}: Initial Spearman={initial_spearman:.4f}, Final Spearman={final_spearman:.4f}")
    else:
        logger.info(f"{dms_id}: Spearman values not available")
    
    return df_logs


def main():
    parser = argparse.ArgumentParser(
        description="Run FGR analysis for ProteinGym fitness prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='fgr_proteingym_config.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--dms_index',
        type=int,
        default=None,
        help='Run only for specific DMS index (for parallel runs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    job_id = os.getenv("SLURM_JOB_ID", str(uuid.uuid4())[:8])
    log_file = output_dir / f"fgr_proteingym_{job_id}.log"
    logger = setup_logging(config['log_level'], log_file)
    
    logger.info("=" * 60)
    logger.info("FGR ProteinGym Fitness Prediction Analysis")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    
    # Load reference file
    data_folder = Path(config['data_folder'])
    reference_file = config['reference_file']
    # Handle absolute vs relative path for reference file
    if Path(reference_file).is_absolute():
        reference_path = Path(reference_file)
    else:
        reference_path = data_folder / reference_file
    
    if not reference_path.exists():
        logger.error(f"Reference file not found: {reference_path}")
        sys.exit(1)
    
    reference_df = pd.read_csv(reference_path)
    logger.info(f"Loaded {len(reference_df)} proteins from reference file")
    
    # Select proteins
    selected_proteins = select_proteins(
        reference_df=reference_df,
        max_length=config['max_sequence_length'],
        num_proteins=config['num_proteins'],
        specific_dms_ids=config.get('specific_dms_ids', []) or None,
    )
    
    # Determine mutation count column
    mutant_count_col = None
    for col in ['DMS_total_number_mutants', 'mutations_total_parsed', 'num_variants_reported']:
        if col in selected_proteins.columns:
            mutant_count_col = col
            break
    
    logger.info(f"Selected {len(selected_proteins)} proteins (max length: {config['max_sequence_length']})")
    for _, row in selected_proteins.iterrows():
        mutations = row.get(mutant_count_col, 'N/A') if mutant_count_col else 'N/A'
        logger.info(f"  - {row['DMS_id']}: length={row['seq_len']}, mutations={mutations}")
    
    # Run experiments
    if args.dms_index is not None:
        # Run single protein
        if args.dms_index >= len(selected_proteins):
            logger.error(f"DMS index {args.dms_index} out of range (max: {len(selected_proteins)-1})")
            sys.exit(1)
        
        dms_row = selected_proteins.iloc[args.dms_index]
        run_protein_experiment(dms_row, config, output_dir, logger)
    else:
        # Run all proteins
        all_results = []
        for i, (_, dms_row) in enumerate(selected_proteins.iterrows()):
            logger.info(f"\n{'='*60}")
            logger.info(f"Protein {i+1}/{len(selected_proteins)}")
            logger.info("=" * 60)
            
            result = run_protein_experiment(dms_row, config, output_dir, logger)
            if result is not None:
                all_results.append(result)
        
        # Save combined results
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(output_dir / "all_proteins_combined.csv", index=False)
            logger.info(f"Saved combined results to {output_dir / 'all_proteins_combined.csv'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
