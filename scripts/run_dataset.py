#!/usr/bin/env python3
"""
Run ProteinTTT on a dataset (single seed, no plotting).

Usage:
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml --output_dir /path/to/output
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml --seed 7 --lr 0.04
"""

import sys
import os
import re
import shutil
import argparse
import yaml
import logging
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import esm
import biotite.structure.io as bsio

from proteinttt.models.esmfold import (
    ESMFoldTTT,
    DEFAULT_ESMFOLD_TTT_CFG,
    GRAD_CLIP_ESMFOLD_TTT_CFG,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msa import generate_msa
from structure_detects import describe_protein_structure


# ---------------------------------------------------------------------------
# Helpers (reused from run_benchmark.py)
# ---------------------------------------------------------------------------

def set_dynamic_chunk_size(model, sequence_length: int) -> int:
    """Dynamically set chunk size based on sequence length."""
    if sequence_length < 200:
        chunk_size = 256
    elif sequence_length < 470:
        chunk_size = 128
    elif sequence_length < 500:
        chunk_size = 32
    elif sequence_length < 600:
        chunk_size = 16
    elif sequence_length < 700:
        chunk_size = 8
    else:
        chunk_size = 4
    # model.set_chunk_size(chunk_size)
    return chunk_size


def ss_percentages(pdb_path: str) -> tuple[float, float, float]:
    """Return (helix%, sheet%, loop%) for the structure at pdb_path."""
    ss = describe_protein_structure(pdb_path)
    n = len(ss)
    if n == 0:
        return (0.0, 0.0, 0.0)
    helix = float(np.sum(ss == 0)) / n * 100.0
    sheet = float(np.sum(ss == 1)) / n * 100.0
    loop = float(np.sum(ss == 2)) / n * 100.0
    return (helix, sheet, loop)


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(model, config, seed, df, output_dir, pdb_dir, msa_dir, job_suffix):
    """Run ProteinTTT on all proteins for one seed. Returns per-protein results DataFrame."""
    logs_root = output_dir / "logs"
    esm_ttt_dir = output_dir / "predicted_structures" / "ESMFold_ProteinTTT"
    esm_dir = output_dir / "predicted_structures" / "ESMFold"
    for d in [logs_root, esm_ttt_dir, esm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model.ttt_cfg.seed = seed

    id_col = config["columns"]["id_column"]
    seq_col = config["columns"]["sequence_column"]
    save_results = config.get("save_results_table", True)
    compute_ss = save_results and config.get("compute_ss_percentages", False)

    results = []
    processed = 0
    total_elapsed = 0.0
    for idx, row in df.iterrows():
        seq_id = str(row[id_col])
        seq = str(row[seq_col]).strip().upper()

        # Skip if this protein has already been processed in a prior run
        if (esm_ttt_dir / f"{seq_id}.pdb").exists():
            logging.info(f"{seq_id}: output already exists, skipping")
            continue

        true_path = pdb_dir / f"{seq_id}.pdb"
        has_reference = true_path.exists()

        if not has_reference:
            logging.info(f"No reference PDB for {seq_id}, will use pLDDT only")

        logging.info(f"Processing {seq_id} (length: {len(seq)})")
        start_time = time.time()
        # Reset RNG state per-protein so identical sequences yield identical TTT outputs
        model.ttt_generator.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        chunk_size = set_dynamic_chunk_size(model, len(seq))
        model.ttt_reset()
        model.set_chunk_size(chunk_size)

        try:
            # Determine MSA file (supports both flat <msa_dir>/<id>.a3m
            # and batched <msa_dir>/<batch>/<id>.a3m layouts)
            msa_file = None
            if config.get("msa", False):
                batch = row["batch"] if "batch" in row.index and pd.notna(row.get("batch")) else None
                msa_file = msa_dir / str(batch) / f"{seq_id}.a3m" if batch else msa_dir / f"{seq_id}.a3m"
                if not msa_file.exists():
                    if config.get("generate_msa", False):
                        logging.info(f"Generating MSA for {seq_id} -> {msa_dir}")
                        try:
                            msa_file = generate_msa(seq, seq_id, cache_dir=msa_dir)
                        except Exception as e:
                            logging.warning(f"MSA generation failed for {seq_id}: {e}, skipping protein")
                            continue
                    else:
                        logging.warning(f"MSA not found ({msa_file}) and generate_msa=false, skipping {seq_id}")
                        continue

            # Run TTT (pass correct_pdb_path only if reference exists)
            ttt_result = model.ttt(
                seq, msa_pth=msa_file,
                correct_pdb_path=true_path if has_reference else None,
            )

            # Save per-step metrics (compact – no PDB strings)
            protein_log_dir = logs_root / seq_id
            protein_log_dir.mkdir(parents=True, exist_ok=True)
            df_logs = ttt_result["df"].copy()
            df_logs.to_csv(protein_log_dir / f"{seq_id}_log.tsv", sep="\t", index=False)

            # Save per-step PDB structures
            step_data = ttt_result["ttt_step_data"]
            for step_idx, step_entry in step_data.items():
                pdb_val = step_entry.get("eval_step_preds", {}).get("pdb")
                if pdb_val is None:
                    continue
                pdb_str = pdb_val[0] if isinstance(pdb_val, list) else pdb_val
                with open(protein_log_dir / f"step_{step_idx}.pdb", "w") as f:
                    f.write(pdb_str)

            # Save before-TTT (step-0) structure
            pdb_before = step_data[0]["eval_step_preds"]["pdb"]
            pdb_str_before = pdb_before[0] if isinstance(pdb_before, list) else pdb_before
            with open(esm_dir / f"{seq_id}.pdb", "w") as f:
                f.write(pdb_str_before)

            # Predict final structure (model is at best-state after ttt())
            with torch.no_grad():
                pdb_str_after = model.infer_pdb(seq)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out_pdb = esm_ttt_dir / f"{seq_id}.pdb"
            with open(out_pdb, "w") as f:
                f.write(pdb_str_after)

            if config.get("describe_structure", False):
                try:
                    description = describe_protein_structure(str(out_pdb))
                    logging.info(f"{seq_id} structure description: {description}")
                except Exception as e:
                    logging.warning(f"describe_protein_structure failed for {seq_id}: {e}")

            elapsed = time.time() - start_time
            processed += 1
            total_elapsed += elapsed

            if save_results:
                # Table metrics — only computed when results.csv will be written
                plddt_before = float(df_logs["plddt"].iloc[0])
                struct = bsio.load_structure(str(out_pdb), extra_fields=["b_factor"])
                plddt_after = float(np.asarray(struct.b_factor, dtype=float).mean())

                ss_cols = {}
                if compute_ss:
                    try:
                        helix_b, sheet_b, loop_b = ss_percentages(str(esm_dir / f"{seq_id}.pdb"))
                    except Exception as e:
                        logging.warning(f"SS computation failed for {seq_id} (before): {e}")
                        helix_b = sheet_b = loop_b = None
                    try:
                        helix_a, sheet_a, loop_a = ss_percentages(str(out_pdb))
                    except Exception as e:
                        logging.warning(f"SS computation failed for {seq_id} (after): {e}")
                        helix_a = sheet_a = loop_a = None
                    ss_cols = {
                        "helix_pct_ESMFold": helix_b,
                        "sheet_pct_ESMFold": sheet_b,
                        "loop_pct_ESMFold": loop_b,
                        "helix_pct_ProteinTTT": helix_a,
                        "sheet_pct_ProteinTTT": sheet_a,
                        "loop_pct_ProteinTTT": loop_a,
                    }

                logging.info(
                    f"{seq_id}: pLDDT {plddt_before:.2f} -> {plddt_after:.2f}, "
                    f"time: {elapsed:.1f}s"
                )
                results.append(
                    {
                        "id": seq_id,
                        "seed": seed,
                        "pLDDT_ESMFold": plddt_before,
                        "pLDDT_ProteinTTT": plddt_after,
                        **ss_cols,
                        "time_seconds": elapsed,
                    }
                )
            else:
                logging.info(f"{seq_id}: done, time: {elapsed:.1f}s")
        except Exception as e:
            logging.error(f"Error for {seq_id}: {e}")
            traceback.print_exc()
            continue

    avg_time = (total_elapsed / processed) if processed else 0.0
    if save_results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / f"results_{job_suffix}.csv", index=False)
        mean_plddt = results_df["pLDDT_ProteinTTT"].mean() if len(results_df) else 0.0
        logging.info(
            f"Done – {processed} proteins, mean pLDDT = {mean_plddt:.2f}, "
            f"avg time per protein = {avg_time:.1f}s"
        )
        return results_df
    logging.info(
        f"Done – {processed} proteins (results table disabled), "
        f"avg time per protein = {avg_time:.1f}s"
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ProteinTTT on a dataset (single seed, no plots).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Seed (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Explicit output directory. If omitted, uses <df_path>/<save_name>/ from config.")
    parser.add_argument("--name", type=str, default=None, help="Optional suffix appended to save_name (e.g. <save_name>_<name>/)")
    parser.add_argument("--start", type=int, default=None, help="Start row index in the CSV (inclusive, 0-based). For sharding the dataset across jobs.")
    parser.add_argument("--end", type=int, default=None, help="End row index in the CSV (exclusive). For sharding the dataset across jobs.")
    # Hyperparameter overrides (all optional; override the YAML config when provided)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--steps", type=int, default=None, help="Number of TTT steps (overrides config)")
    parser.add_argument("--ags", type=int, default=None, help="Gradient accumulation steps (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank (overrides config)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (overrides config)")
    parser.add_argument("--gradient_clip_max_norm", type=float, default=None, help="Gradient clip max norm (overrides config)")
    parser.add_argument("--confidence_collapse_ratio", type=float, default=None, help="Confidence collapse ratio (overrides config)")
    parser.add_argument("--confidence_collapse_patience", type=int, default=None, help="Confidence collapse patience (overrides config)")
    parser.add_argument("--msa_sampling_strategy", type=str, default=None, help="MSA sampling strategy (overrides config)")
    parser.add_argument("--msa", action="store_true", default=None, help="Enable MSA (overrides config)")
    parser.add_argument("--no_msa", action="store_true", help="Disable MSA (overrides config)")
    parser.add_argument("--gradient_clip", action="store_true", default=None, help="Enable gradient clipping (overrides config)")
    parser.add_argument("--no_gradient_clip", action="store_true", help="Disable gradient clipping (overrides config)")
    parser.add_argument("--compute_ss_percentages", action="store_true", default=None, help="Include helix/sheet/loop % columns in results.csv (overrides config)")
    parser.add_argument("--no_compute_ss_percentages", action="store_true", help="Disable SS % columns (overrides config)")
    parser.add_argument("--save_results_table", action="store_true", default=None, help="Save per-protein results.csv (overrides config)")
    parser.add_argument("--no_save_results_table", action="store_true", help="Skip writing results.csv; only keep logs & predictions (overrides config)")
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Max sequence length (overrides config)")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer: sgd or adamw (overrides config)")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum (overrides config)")
    parser.add_argument("--mask_ratio", type=float, default=None, help="Mask ratio for MLM (overrides config)")
    parser.add_argument("--lr_scheduler", type=str, default=None, help="LR scheduler: cosine, cosine_warmup (overrides config)")
    parser.add_argument("--lr_warmup_steps", type=int, default=None, help="Warmup steps for LR scheduler (overrides config)")
    parser.add_argument("--lr_min", type=float, default=None, help="Minimum LR for scheduler (overrides config)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI hyperparameter overrides
    _hparam_overrides = {
        "lr": args.lr,
        "steps": args.steps,
        "ags": args.ags,
        "batch_size": args.batch_size,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "gradient_clip_max_norm": args.gradient_clip_max_norm,
        "confidence_collapse_ratio": args.confidence_collapse_ratio,
        "confidence_collapse_patience": args.confidence_collapse_patience,
        "msa_sampling_strategy": args.msa_sampling_strategy,
        "max_sequence_length": args.max_sequence_length,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "mask_ratio": args.mask_ratio,
        "lr_scheduler": args.lr_scheduler,
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_min": args.lr_min,
    }
    for key, val in _hparam_overrides.items():
        if val is not None:
            config[key] = val
            print(f"[CLI override] {key} = {val}")
    if args.no_msa:
        config["msa"] = False
        print("[CLI override] msa = False")
    elif args.msa:
        config["msa"] = True
        print("[CLI override] msa = True")
    if args.no_gradient_clip:
        config["gradient_clip"] = False
        print("[CLI override] gradient_clip = False")
    elif args.gradient_clip:
        config["gradient_clip"] = True
        print("[CLI override] gradient_clip = True")
    if args.no_compute_ss_percentages:
        config["compute_ss_percentages"] = False
        print("[CLI override] compute_ss_percentages = False")
    elif args.compute_ss_percentages:
        config["compute_ss_percentages"] = True
        print("[CLI override] compute_ss_percentages = True")
    if args.no_save_results_table:
        config["save_results_table"] = False
        print("[CLI override] save_results_table = False")
    elif args.save_results_table:
        config["save_results_table"] = True
        print("[CLI override] save_results_table = True")

    # Force per-step metrics so logs are populated
    config["compute_step_metrics"] = True

    seed = args.seed if args.seed is not None else config.get("seed", 0)

    # Paths
    source_base_path = Path(config["df_path"]).expanduser().resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        save_name = config["input"].get("save_name")
        if not save_name:
            print("Either --output_dir or config['input']['save_name'] must be provided.")
            sys.exit(1)
        run_name = f"{save_name}_{args.name}" if args.name else save_name
        run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        output_dir = source_base_path / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config into the experiment directory for reproducibility
    shutil.copy2(config_path, output_dir / "config.yaml")

    pdb_dir = Path(config["input"]["pdb_dir"])
    msa_dir = Path(config["input"]["msa_dir"])
    summary_path = source_base_path / config["input"]["summary_file"]

    job_suffix = os.getenv("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / f"dataset_{job_suffix}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    # Prevent double logging from the proteinttt library
    logging.getLogger("ttt_log").propagate = False

    logging.info(f"Config: {config_path}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Output: {output_dir}")

    # Load data
    df = pd.read_csv(summary_path)
    total_rows = len(df)
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else total_rows
        df = df.iloc[start:end].copy()
        logging.info(f"Sharding CSV rows [{start}, {end}) of {total_rows} -> {len(df)} proteins")
    df["sequence_length"] = df[config["columns"]["sequence_column"]].apply(len)
    max_len = config.get("max_sequence_length", 500)
    df = df.query(f"sequence_length <= {max_len}").copy()
    logging.info(f"Loaded {len(df)} proteins (max length: {max_len})")

    # Load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    base_model = esm.pretrained.esmfold_v0().eval().to(device)

    ttt_cfg = GRAD_CLIP_ESMFOLD_TTT_CFG if config.get("gradient_clip", False) else DEFAULT_ESMFOLD_TTT_CFG
    SCRIPT_ONLY_KEYS = {"df_path", "output", "input", "compute_step_metrics", "new_experement_dir", "columns", "generate_msa", "describe_structure", "compute_ss_percentages", "save_results_table"}
    for key, value in config.items():
        if key not in SCRIPT_ONLY_KEYS:
            setattr(ttt_cfg, key, value)

    if config.get("msa", False):
        base_model.set_chunk_size(128)

    logging.info(f"TTT config: {ttt_cfg}")

    model = ESMFoldTTT.ttt_from_pretrained(
        base_model, ttt_cfg=ttt_cfg, esmfold_config=base_model.cfg
    ).to(device)

    # Run dataset
    total_start = time.time()
    run_dataset(model, config, seed, df, output_dir, pdb_dir, msa_dir, job_suffix)
    total_time = time.time() - total_start
    logging.info(f"Total runtime: {total_time:.1f}s ({total_time / 3600:.1f}h)")
    logging.info("Dataset run complete!")


if __name__ == "__main__":
    main()
