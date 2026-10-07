#!/usr/bin/env python3
"""
Run ProteinTTT on a dataset (one or more seeds, no plotting).

Usage:
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml --output_dir /path/to/output
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml --seed 7 --lr 0.04
    python scripts/run_dataset.py --config scripts/config_benchmark.yaml --seed 1,2,3

With several seeds each seed gets its own <output_dir>/seed_<N>/ subdirectory
(logs, predicted structures, results_<job>.csv), plus a combined
<output_dir>/results_<job>.csv across all seeds.

With `rerun_helix: true` the structure ProteinTTT selected is tested for the
one-helix (collapsed-rod) artifact; a protein that fails the test is run again
with seed + `rerun_seed_offset` and only that second run is kept on disk.
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
import biotite.structure.io as bsio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_msa import generate_msa
from structure_detects import describe_protein_structure
from add_helix_filter import (
    HELIX_DOMINANCE_MIN,
    HELIX_PCT_MIN,
    PLDDT_CONFIDENT,
    ss_features,
)


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


def parse_seeds(value) -> list[int]:
    """Normalise a seed spec into a list of ints.

    Accepts an int (``1``), a YAML list (``[1, 2, 3]``) or a comma/space
    separated string (``"1, 2, 3"``).
    """
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, str):
        items = [tok for tok in re.split(r"[,\s]+", value.strip()) if tok]
    else:
        items = [value]
    seeds = [int(item) for item in items]
    if not seeds:
        raise ValueError(f"No seeds parsed from {value!r}")
    return seeds


def mean_plddt(pdb_path: Path) -> float:
    """Mean pLDDT, read from the B-factor column of a predicted structure."""
    struct = bsio.load_structure(str(pdb_path), extra_fields=["b_factor"])
    return float(np.asarray(struct.b_factor, dtype=float).mean())


def one_helix_check(pdb_path: Path, plddt: float, helix_min: float,
                    dominance_min: float, plddt_min: float) -> tuple[bool, dict[str, float]]:
    """Is this structure a confident collapsed rod? Returns (flag, helix features).

    Same two-part test as `one_helix_flag` in scripts/add_helix_filter.py:
    most of the chain is helix AND most of that helix is a single segment.
    """
    feats = ss_features(pdb_path)
    if not feats:
        return False, {}
    flag = (
        feats["helix_pct"] >= helix_min
        and feats["helix_dominance"] >= dominance_min
        and plddt >= plddt_min
    )
    return flag, feats


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

    # One-helix rerun: if the structure ProteinTTT selected is a confident
    # collapsed rod, run the protein once more with a different seed and keep
    # only that second run.
    rerun_helix = config.get("rerun_helix", False)
    rerun_seed_offset = int(config.get("rerun_seed_offset", 1000))
    helix_min = float(config.get("helix_pct_min", HELIX_PCT_MIN))
    dominance_min = float(config.get("helix_dominance_min", HELIX_DOMINANCE_MIN))
    helix_plddt_min = float(config.get("helix_plddt_min", PLDDT_CONFIDENT))
    if rerun_helix:
        logging.info(
            f"One-helix rerun enabled: helix_pct >= {helix_min:g}, "
            f"helix_dominance >= {dominance_min:g}, pLDDT >= {helix_plddt_min:g}; "
            f"rerun seed = seed + {rerun_seed_offset}"
        )

    results = []
    processed = 0
    n_rerun = 0
    total_elapsed = 0.0
    for idx, row in df.iterrows():
        seq_id = str(row[id_col])
        seq = str(row[seq_col]).strip().upper()

        # Skip if this protein has already been processed in a prior run
        if (esm_dir / f"{seq_id}.pdb").exists():
            logging.info(f"{seq_id}: output already exists, skipping")
            continue

        true_path = pdb_dir / f"{seq_id}.pdb"
        has_reference = true_path.exists()

        if not has_reference:
            logging.info(f"No reference PDB for {seq_id}, will use pLDDT only")

        logging.info(f"Processing {seq_id} (length: {len(seq)})")
        start_time = time.time()
        chunk_size = set_dynamic_chunk_size(model, len(seq))

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

            def ttt_pass(pass_seed: int) -> tuple[pd.DataFrame, str]:
                """One complete TTT pass at pass_seed.

                Writes the per-step logs/PDBs and the before-TTT baseline,
                overwriting anything a previous pass left behind, and returns
                (per-step log frame, final predicted PDB string).
                """
                # Reset RNG state per pass so identical sequences yield identical TTT outputs
                model.ttt_generator.manual_seed(pass_seed)
                torch.manual_seed(pass_seed)
                np.random.seed(pass_seed)
                model.ttt_cfg.seed = pass_seed
                model.ttt_reset()
                model.set_chunk_size(chunk_size)

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

                # Save per-step PDB structures. Drop stale ones first: a rerun may
                # early-stop at a different step, and only its own trace should remain.
                for stale in protein_log_dir.glob("step_*.pdb"):
                    stale.unlink()
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
                return df_logs, pdb_str_after

            out_pdb = esm_ttt_dir / f"{seq_id}.pdb"
            df_logs, pdb_str_after = ttt_pass(seed)
            with open(out_pdb, "w") as f:
                f.write(pdb_str_after)

            plddt_before = float(df_logs["plddt"].iloc[0])
            plddt_after = mean_plddt(out_pdb) if (save_results or rerun_helix) else None

            # One-helix rerun. Only the structure ProteinTTT selected is tested,
            # and a triggered rerun replaces the first run entirely.
            rerun = False
            rerun_seed = None
            plddt_first = None
            one_helix, helix_feats = False, {}
            if rerun_helix:
                one_helix, helix_feats = one_helix_check(
                    out_pdb, plddt_after, helix_min, dominance_min, helix_plddt_min
                )
                if one_helix:
                    rerun_seed = seed + rerun_seed_offset
                    plddt_first = plddt_after
                    logging.info(
                        f"{seq_id}: one-helix rod (helix {helix_feats['helix_pct']:.1f}%, "
                        f"dominance {helix_feats['helix_dominance']:.1f}%, "
                        f"pLDDT {plddt_after:.2f}) -> rerun with seed {rerun_seed}"
                    )
                    df_logs, pdb_str_after = ttt_pass(rerun_seed)
                    with open(out_pdb, "w") as f:
                        f.write(pdb_str_after)
                    plddt_before = float(df_logs["plddt"].iloc[0])
                    plddt_after = mean_plddt(out_pdb)
                    one_helix, helix_feats = one_helix_check(
                        out_pdb, plddt_after, helix_min, dominance_min, helix_plddt_min
                    )
                    rerun = True
                    n_rerun += 1
                    logging.info(
                        f"{seq_id}: rerun pLDDT {plddt_first:.2f} -> {plddt_after:.2f}, "
                        f"still one-helix: {one_helix}"
                    )

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

                # Rerun bookkeeping. Only emitted when the feature is on, so the
                # table schema is unchanged for rerun_helix: false.
                rerun_cols = {}
                if rerun_helix:
                    rerun_cols = {
                        "helix_pct_ProteinTTT": helix_feats.get("helix_pct"),
                        "helix_dominance_ProteinTTT": helix_feats.get("helix_dominance"),
                        "one_helix_flag": one_helix,
                        "rerun": rerun,
                        "rerun_seed": rerun_seed,
                        "pLDDT_ProteinTTT_first": plddt_first,
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
                        **rerun_cols,
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
    if rerun_helix:
        logging.info(f"One-helix reruns: {n_rerun} / {processed} proteins")
    if save_results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / f"results_{job_suffix}.csv", index=False)
        mean_plddt_ttt = results_df["pLDDT_ProteinTTT"].mean() if len(results_df) else 0.0
        logging.info(
            f"Done – {processed} proteins, mean pLDDT = {mean_plddt_ttt:.2f}, "
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
    parser.add_argument("--seed", type=str, default=None,
                        help="Seed, or several comma-separated seeds e.g. '1,2,3' (overrides config)")
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
    parser.add_argument("--rerun_helix", action="store_true", default=None, help="Rerun a protein with a different seed when ProteinTTT's chosen structure is a one-helix rod (overrides config)")
    parser.add_argument("--no_rerun_helix", action="store_true", help="Disable the one-helix rerun (overrides config)")
    parser.add_argument("--rerun_seed_offset", type=int, default=None, help="Rerun seed = seed + this offset (overrides config)")
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
        "rerun_seed_offset": args.rerun_seed_offset,
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
    if args.no_rerun_helix:
        config["rerun_helix"] = False
        print("[CLI override] rerun_helix = False")
    elif args.rerun_helix:
        config["rerun_helix"] = True
        print("[CLI override] rerun_helix = True")
    if args.no_save_results_table:
        config["save_results_table"] = False
        print("[CLI override] save_results_table = False")
    elif args.save_results_table:
        config["save_results_table"] = True
        print("[CLI override] save_results_table = True")

    # Force per-step metrics so logs are populated
    config["compute_step_metrics"] = True

    seeds = parse_seeds(args.seed if args.seed is not None else config.get("seed", 0))

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
    logging.info(f"Seeds: {seeds}")
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

    if config.get("model", "esmfold") == "esmfold2":
        from proteinttt.models.esmfold2 import ESMFold2TTT, load_esmfold2, DEFAULT_ESMFOLD2_TTT_CFG
        ttt_cfg = DEFAULT_ESMFOLD2_TTT_CFG
    else:
        import esm
        from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG, GRAD_CLIP_ESMFOLD_TTT_CFG
        base_model = esm.pretrained.esmfold_v0().eval().to(device)
        ttt_cfg = GRAD_CLIP_ESMFOLD_TTT_CFG if config.get("gradient_clip", False) else DEFAULT_ESMFOLD_TTT_CFG
    # "seed" is excluded: run_dataset() sets ttt_cfg.seed per seed itself, and the
    # config value may be a list.
    SCRIPT_ONLY_KEYS = {"df_path", "output", "input", "compute_step_metrics", "new_experement_dir", "columns", "generate_msa", "describe_structure", "compute_ss_percentages", "save_results_table", "seed", "rerun_helix", "rerun_seed_offset", "helix_pct_min", "helix_dominance_min", "helix_plddt_min", "model", "esmfold2_models"}
    for key, value in config.items():
        if key not in SCRIPT_ONLY_KEYS:
            setattr(ttt_cfg, key, value)

    logging.info(f"TTT config: {ttt_cfg}")

    if config.get("model", "esmfold") == "esmfold2":
        model = ESMFold2TTT(
            ttt_cfg, *(m.to(device) for m in load_esmfold2(**config.get("esmfold2_models", {})))
        )
    else:
        if config.get("msa", False):
            base_model.set_chunk_size(128)
        model = ESMFoldTTT.ttt_from_pretrained(
            base_model, ttt_cfg=ttt_cfg, esmfold_config=base_model.cfg
        ).to(device)

    # Run dataset once per seed. With several seeds each gets its own
    # subdirectory so outputs (and the already-processed skip check) don't collide.
    total_start = time.time()
    per_seed_results = []
    for seed in seeds:
        seed_output_dir = output_dir / f"seed_{seed}" if len(seeds) > 1 else output_dir
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"=== Seed {seed} -> {seed_output_dir} ===")
        seed_start = time.time()
        results_df = run_dataset(
            model, config, seed, df, seed_output_dir, pdb_dir, msa_dir, job_suffix
        )
        logging.info(f"Seed {seed} finished in {time.time() - seed_start:.1f}s")
        if results_df is not None and len(results_df):
            per_seed_results.append(results_df)

    # Combined table across all seeds
    if len(seeds) > 1 and per_seed_results:
        combined = pd.concat(per_seed_results, ignore_index=True)
        combined_path = output_dir / f"results_{job_suffix}.csv"
        combined.to_csv(combined_path, index=False)
        logging.info(f"Combined results across {len(seeds)} seeds -> {combined_path}")

    total_time = time.time() - total_start
    logging.info(f"Total runtime: {total_time:.1f}s ({total_time / 3600:.1f}h)")
    logging.info("Dataset run complete!")


if __name__ == "__main__":
    main()
