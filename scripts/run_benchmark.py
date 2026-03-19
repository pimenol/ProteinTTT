#!/usr/bin/env python3
"""
Benchmark ProteinTTT: Run with multiple seeds, compute per-step LDDT, produce plots.

Usage:
    python scripts/run_benchmark.py --config scripts/config_benchmark.yaml
    python scripts/run_benchmark.py --config scripts/config_benchmark.yaml --seeds 0 1 2
    python scripts/run_benchmark.py --config scripts/config_benchmark.yaml --output_dir /path/to/output
    python scripts/run_benchmark.py --plots_only --output_dir /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/data/bfvd/experements_msa/best/benchmark/lr_0.04_ags_8_msa_True_grad_1.0_random_3998958 --seeds 0 1 2
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
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import esm
import biotite.structure.io as bsio

from proteinttt.models.esmfold import (
    ESMFoldTTT,
    DEFAULT_ESMFOLD_TTT_CFG,
    GRAD_CLIP_ESMFOLD_TTT_CFG,
)
from proteinttt.utils.structure import lddt_score


# ---------------------------------------------------------------------------
# Helpers (reused from run_df.py)
# ---------------------------------------------------------------------------

def set_dynamic_chunk_size(model, sequence_length):
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
    model.set_chunk_size(chunk_size)
    return chunk_size


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def run_seed(model, config, seed, df, output_dir, pdb_dir, msa_dir):
    """Run ProteinTTT on all proteins for one seed. Returns per-protein results DataFrame."""
    seed_dir = output_dir / f"seed_{seed}"
    logs_dir = seed_dir / "logs"
    esm_ttt_dir = seed_dir / "predicted_structures" / "ESMFold_ProteinTTT"
    esm_dir = seed_dir / "predicted_structures" / "ESMFold"
    for d in [logs_dir, esm_ttt_dir, esm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Set seed everywhere
    model.ttt_cfg.seed = seed
    model.ttt_generator.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    id_col = config["columns"]["id_column"]
    seq_col = config["columns"]["sequence_column"]

    results = []
    for idx, row in df.iterrows():
        seq_id = str(row[id_col])
        seq = str(row[seq_col]).strip().upper()
        true_path = pdb_dir / f"{seq_id}.pdb"

        if not true_path.exists():
            logging.warning(f"[Seed {seed}] Reference PDB not found: {true_path}, skipping {seq_id}")
            continue

        logging.info(f"[Seed {seed}] Processing {seq_id} (length: {len(seq)})")
        start_time = time.time()
        chunk_size = set_dynamic_chunk_size(model, len(seq))
        model.ttt_reset()
        model.set_chunk_size(chunk_size)

        try:
            # Determine MSA file
            msa_file = None
            if config.get("msa", False):
                msa_file = msa_dir / f"{seq_id}.a3m"
                if not msa_file.exists():
                    logging.warning(f"[Seed {seed}] MSA not found: {msa_file}, running without MSA")
                    msa_file = None

            # Run TTT with per-step LDDT
            ttt_result = model.ttt(seq, msa_pth=msa_file, correct_pdb_path=true_path)

            # Save per-step metrics (compact – no PDB strings)
            df_logs = ttt_result["df"].copy()
            df_logs.to_csv(logs_dir / f"{seq_id}_log.tsv", sep="\t", index=False)

            # Save before-TTT (step-0) structure
            step_data = ttt_result["ttt_step_data"]
            pdb_before = step_data[0]["eval_step_preds"]["pdb"]
            pdb_str_before = pdb_before[0] if isinstance(pdb_before, list) else pdb_before
            with open(esm_dir / f"{seq_id}.pdb", "w") as f:
                f.write(pdb_str_before)
            plddt_before = float(df_logs["plddt"].iloc[0])

            # Predict final structure (model is at best-state after ttt())
            with torch.no_grad():
                pdb_str_after = model.infer_pdb(seq)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out_pdb = esm_ttt_dir / f"{seq_id}.pdb"
            with open(out_pdb, "w") as f:
                f.write(pdb_str_after)
            struct = bsio.load_structure(str(out_pdb), extra_fields=["b_factor"])
            plddt_after = float(np.asarray(struct.b_factor, dtype=float).mean())

            # Compute final LDDT against reference
            lddt_after = lddt_score(str(true_path), str(out_pdb))
            lddt_before = lddt_score(str(true_path), str(esm_dir / f"{seq_id}.pdb"))

            elapsed = time.time() - start_time
            logging.info(
                f"[Seed {seed}] {seq_id}: pLDDT {plddt_before:.2f} -> {plddt_after:.2f}, "
                f"LDDT {lddt_before:.4f} -> {lddt_after:.4f}, time: {elapsed:.1f}s"
            )

            results.append(
                {
                    "id": seq_id,
                    "seed": seed,
                    "pLDDT_ESMFold": plddt_before,
                    "pLDDT_ProteinTTT": plddt_after,
                    "lddt_ESMFold": lddt_before,
                    "lddt_ProteinTTT": lddt_after,
                    "time_seconds": elapsed,
                }
            )
        except Exception as e:
            logging.error(f"[Seed {seed}] Error for {seq_id}: {e}")
            traceback.print_exc()
            continue

    # Save per-seed summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(seed_dir / "results.tsv", sep="\t", index=False)

    avg_time = results_df["time_seconds"].mean() if len(results_df) else 0.0
    logging.info(
        f"[Seed {seed}] Done – {len(results_df)} proteins, "
        f"mean LDDT = {results_df['lddt_ProteinTTT'].mean():.4f}, "
        f"avg time per protein = {avg_time:.1f}s"
    )
    return results_df


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_plots(output_dir, seeds):
    """Read logs & results for all seeds and produce three plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0"]

    # ---- Collect per-step data per seed ----
    step_data_per_seed = {}
    for seed in seeds:
        logs_dir = output_dir / f"seed_{seed}" / "logs"
        if not logs_dir.exists():
            logging.warning(f"Logs directory not found for seed {seed}")
            continue
        all_dfs = []
        for log_file in sorted(logs_dir.glob("*_log.tsv")):
            try:
                ldf = pd.read_csv(log_file, sep="\t")
                ldf["protein_id"] = log_file.stem.replace("_log", "")
                all_dfs.append(ldf)
            except Exception as e:
                logging.warning(f"Error reading {log_file}: {e}")
        if all_dfs:
            step_data_per_seed[seed] = pd.concat(all_dfs, ignore_index=True)

    # ---- Collect final results per seed ----
    final_results = {}
    for seed in seeds:
        rf = output_dir / f"seed_{seed}" / "results.tsv"
        if rf.exists():
            final_results[seed] = pd.read_csv(rf, sep="\t")

    if not final_results:
        logging.error("No results found for any seed – cannot generate plots.")
        return

    # ======================================================================
    # Plot 1: Mean LDDT per step
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    all_mean_lddts = []
    for i, seed in enumerate(seeds):
        if seed not in step_data_per_seed:
            continue
        sdf = step_data_per_seed[seed]
        if "lddt" not in sdf.columns or sdf["lddt"].dropna().empty:
            logging.warning(f"No per-step LDDT for seed {seed}")
            continue
        mean_lddt = sdf.groupby("step")["lddt"].mean()
        ax.plot(
            mean_lddt.index, mean_lddt.values,
            "o-", color=colors[i % len(colors)],
            alpha=0.5, linewidth=1, markersize=3, label=f"Seed {seed}",
        )
        all_mean_lddts.append(mean_lddt)

    if all_mean_lddts:
        common_steps = all_mean_lddts[0].index
        for ml in all_mean_lddts[1:]:
            common_steps = common_steps.intersection(ml.index)
        aligned = np.array([[ml[s] for s in common_steps] for ml in all_mean_lddts])
        mean_vals = aligned.mean(axis=0)
        std_vals = aligned.std(axis=0)
        ax.plot(
            common_steps, mean_vals,
            "o-", color="black", linewidth=2.5, markersize=5, label="Mean", zorder=5,
        )
        ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color="gray")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean LDDT", fontsize=12)
    ax.set_title("Mean LDDT per Step", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_lddt_per_step.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ======================================================================
    # Plot 2: Mean pLDDT per step
    # ======================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    all_mean_plddts = []
    for i, seed in enumerate(seeds):
        if seed not in step_data_per_seed:
            continue
        sdf = step_data_per_seed[seed]
        mean_plddt = sdf.groupby("step")["plddt"].mean()
        ax.plot(
            mean_plddt.index, mean_plddt.values,
            "o-", color=colors[i % len(colors)],
            alpha=0.5, linewidth=1, markersize=3, label=f"Seed {seed}",
        )
        all_mean_plddts.append(mean_plddt)

    if all_mean_plddts:
        common_steps = all_mean_plddts[0].index
        for ml in all_mean_plddts[1:]:
            common_steps = common_steps.intersection(ml.index)
        aligned = np.array([[ml[s] for s in common_steps] for ml in all_mean_plddts])
        mean_vals = aligned.mean(axis=0)
        std_vals = aligned.std(axis=0)
        ax.plot(
            common_steps, mean_vals,
            "o-", color="black", linewidth=2.5, markersize=5, label="Mean", zorder=5,
        )
        ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color="gray")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean pLDDT", fontsize=12)
    ax.set_title("Mean pLDDT per Step", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_plddt_per_step.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ======================================================================
    # Plot 3: Bar plot – Mean LDDT and Mean Time per protein (across seeds)
    # ======================================================================
    seed_mean_lddts = []
    seed_mean_times = []
    valid_seeds = []
    for seed in seeds:
        if seed in final_results:
            seed_mean_lddts.append(final_results[seed]["lddt_ProteinTTT"].mean())
            seed_mean_times.append(final_results[seed]["time_seconds"].mean())
            valid_seeds.append(seed)

    if not seed_mean_lddts:
        logging.error("No final LDDT data available for bar plot.")
        return

    lddt_mean = np.mean(seed_mean_lddts)
    lddt_std = np.std(seed_mean_lddts)
    time_mean = np.mean(seed_mean_times)
    time_std = np.std(seed_mean_times)
    n_proteins = len(final_results[valid_seeds[0]])

    fig, ax1 = plt.subplots(figsize=(7, 6))

    x = np.array([0, 1])
    bar_colors = ["#2196F3", "#FF9800"]
    point_color = "#4CAF50"

    # Semi-transparent bars
    ax1.bar(x[0], lddt_mean, width=0.5, color=bar_colors[0],
            alpha=0.35, edgecolor="black", linewidth=0.5, label="Mean LDDT")
    ax1.errorbar(x[0], lddt_mean, yerr=lddt_std, fmt="none", ecolor="black", capsize=6, linewidth=2)

    # Scatter individual seed values on top (like the reference plot style)
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(seed_mean_lddts))
    ax1.scatter(np.zeros(len(seed_mean_lddts)) + jitter, seed_mean_lddts,
                color=point_color, edgecolors="black", linewidths=0.5,
                s=60, zorder=5, label="Per-seed")

    ax1.set_ylabel("Mean LDDT", fontsize=12, color=bar_colors[0])
    ax1.tick_params(axis="y", labelcolor=bar_colors[0])

    # Second y-axis for time
    ax2 = ax1.twinx()
    ax2.bar(x[1], time_mean, width=0.5, color=bar_colors[1],
            alpha=0.35, edgecolor="black", linewidth=0.5, label="Mean Time")
    ax2.errorbar(x[1], time_mean, yerr=time_std, fmt="none", ecolor="black", capsize=6, linewidth=2)

    # Scatter individual seed values for time
    jitter_t = np.random.default_rng(42).uniform(-0.12, 0.12, len(seed_mean_times))
    ax2.scatter(np.ones(len(seed_mean_times)) + jitter_t, seed_mean_times,
                color=point_color, edgecolors="black", linewidths=0.5,
                s=60, zorder=5)

    ax2.set_ylabel("Mean Time per Protein (s)", fontsize=12, color=bar_colors[1])
    ax2.tick_params(axis="y", labelcolor=bar_colors[1])

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Mean LDDT", "Mean Time"], fontsize=11)
    ax1.set_title(f"Benchmark Summary ({n_proteins} proteins, {len(valid_seeds)} seeds)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # Annotate bars
    ax1.text(0, lddt_mean + lddt_std + 0.008,
             f"{lddt_mean:.4f} ± {lddt_std:.4f}",
             ha="center", va="bottom", fontsize=10, fontweight="bold", color=bar_colors[0])
    ax2.text(1, time_mean + time_std + 0.5,
             f"{time_mean:.1f} ± {time_std:.1f}s",
             ha="center", va="bottom", fontsize=10, fontweight="bold", color=bar_colors[1])

    plt.tight_layout()
    plt.savefig(plots_dir / "final_mean_lddt_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Plots saved to {plots_dir}")
    logging.info(f"  Mean LDDT across seeds: {lddt_mean:.4f} ± {lddt_std:.4f}")
    logging.info(f"  Mean time per protein:  {time_mean:.1f}s ± {time_std:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ProteinTTT with multiple seeds and produce aggregated plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds (default: 0 1 2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Explicit output directory. If omitted, a unique dir is created under <df_path>/benchmark/")
    parser.add_argument("--name", type=str, default=None, help="Custom experiment name (used in auto-generated dir)")
    parser.add_argument("--plots_only", action="store_true", help="Skip TTT runs, only regenerate plots from existing data")
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

    # Force per-step metrics for benchmark
    config["compute_step_metrics"] = True

    # Paths
    source_base_path = Path(config["df_path"]).expanduser().resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-generate a unique experiment directory from ALL hyperparameters
        job_suffix = os.getenv("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))

        if args.name:
            run_name = f"{args.name}_{job_suffix}"
        else:
            # Keys that are structural / not hyperparameters — skip them in the name
            _SKIP_NAME_KEYS = {
                "df_path", "output", "input", "columns", "seed",
                "compute_step_metrics", "new_experement_dir",
                "use_true_pdb", "use_chain_name", "renumber_pdb",
                "max_sequence_length",
            }
            # Short aliases for compact directory names
            _NAME_ALIAS = {
                "lr": "lr", "ags": "ags", "msa": "msa", "steps": "s",
                "lora_rank": "r", "lora_alpha": "a",
                "gradient_clip": "clip", "gradient_clip_max_norm": "gnorm",
                "msa_sampling_strategy": "strat",
                "confidence_collapse_ratio": "ccr",
                "confidence_collapse_patience": "ccp",
                "optimizer": "opt", "momentum": "mom",
                "mask_ratio": "mr", "lr_scheduler": "lrs",
                "lr_warmup_steps": "warm", "lr_min": "lrmin",
            }

            name_parts = []
            for key in sorted(config.keys()):
                if key in _SKIP_NAME_KEYS or isinstance(config[key], dict):
                    continue
                alias = _NAME_ALIAS.get(key, key)
                val = config[key]
                # Format value compactly
                if isinstance(val, bool):
                    val_str = "1" if val else "0"
                elif isinstance(val, float):
                    val_str = f"{val:g}"
                else:
                    val_str = str(val)
                name_parts.append(f"{alias}{val_str}")
            run_name = "_".join(name_parts) + f"_{job_suffix}"

        run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        output_dir = source_base_path / "benchmark_20" / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config into the experiment directory for reproducibility
    shutil.copy2(config_path, output_dir / "config.yaml")

    pdb_dir = Path(config["input"]["pdb_dir"])
    msa_dir = Path(config["input"]["msa_dir"])
    summary_path = source_base_path / config["input"]["summary_file"]

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "benchmark.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    # Prevent double logging from the proteinttt library
    logging.getLogger("ttt_log").propagate = False

    logging.info(f"Config: {config_path}")
    logging.info(f"Seeds: {args.seeds}")
    logging.info(f"Output: {output_dir}")

    if args.plots_only:
        logging.info("--plots_only: skipping TTT runs, regenerating plots")
        generate_plots(output_dir, args.seeds)
        return

    # Load data
    df = pd.read_csv(summary_path)
    df["sequence_length"] = df[config["columns"]["sequence_column"]].apply(len)
    max_len = config.get("max_sequence_length", 500)
    df = df.query(f"sequence_length <= {max_len}").copy()
    logging.info(f"Loaded {len(df)} proteins (max length: {max_len})")

    # Load model once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    base_model = esm.pretrained.esmfold_v0().eval().to(device)

    ttt_cfg = GRAD_CLIP_ESMFOLD_TTT_CFG if config.get("gradient_clip", False) else DEFAULT_ESMFOLD_TTT_CFG
    SCRIPT_ONLY_KEYS = {"df_path", "output", "input", "compute_step_metrics", "new_experement_dir", "columns"}
    for key, value in config.items():
        if key not in SCRIPT_ONLY_KEYS:
            setattr(ttt_cfg, key, value)

    if config.get("msa", False):
        base_model.set_chunk_size(128)

    logging.info(f"TTT config: {ttt_cfg}")

    model = ESMFoldTTT.ttt_from_pretrained(
        base_model, ttt_cfg=ttt_cfg, esmfold_config=base_model.cfg
    ).to(device)

    # Run benchmark for each seed
    total_start = time.time()
    for seed in args.seeds:
        logging.info("=" * 60)
        logging.info(f"Starting seed {seed}")
        logging.info("=" * 60)
        run_seed(model, config, seed, df, output_dir, pdb_dir, msa_dir)

    total_time = time.time() - total_start
    logging.info(f"Total benchmark time: {total_time:.1f}s ({total_time / 3600:.1f}h)")

    # Generate plots
    generate_plots(output_dir, args.seeds)
    logging.info("Benchmark complete!")


if __name__ == "__main__":
    main()
