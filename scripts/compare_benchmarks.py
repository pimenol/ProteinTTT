#!/usr/bin/env python3
"""
Compare multiple ProteinTTT benchmark experiments.

Reads results from experiment directories created by run_benchmark.py and
produces side-by-side comparison plots plus a summary table.

Usage:
    # Compare specific experiment directories
    python scripts/compare_benchmarks.py \
        path/to/benchmark/exp1 \
        path/to/benchmark/exp2

    # Compare ALL experiments under a benchmark root
    python scripts/compare_benchmarks.py --benchmark_dir /scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/data/benchmark/benchmark_20

    # Custom output location
    python scripts/compare_benchmarks.py --benchmark_dir path/to/benchmark --output_dir path/to/comparison
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(exp_dir):
    """Load results and config from a single experiment directory.

    Returns dict with keys: name, config, seeds, results (list of DataFrames),
    step_data (dict seed -> DataFrame), mean_lddt, std_lddt, mean_time, std_time.
    Returns None if the directory is not a valid experiment.
    """
    exp_dir = Path(exp_dir)

    # Discover seed directories
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    seeds = []
    results = []
    step_data = {}

    for sd in seed_dirs:
        rf = sd / "results.tsv"
        if not rf.exists():
            continue
        seed = int(sd.name.split("_")[1])
        seeds.append(seed)
        results.append(pd.read_csv(rf, sep="\t"))

        # Load per-step logs
        logs_dir = sd / "logs"
        if logs_dir.exists():
            all_dfs = []
            for lf in sorted(logs_dir.glob("*_log.tsv")):
                try:
                    ldf = pd.read_csv(lf, sep="\t")
                    ldf["protein_id"] = lf.stem.replace("_log", "")
                    all_dfs.append(ldf)
                except Exception:
                    pass
            if all_dfs:
                step_data[seed] = pd.concat(all_dfs, ignore_index=True)

    if not results:
        return None

    # Aggregate across seeds
    seed_mean_lddts = [df["lddt_ProteinTTT"].mean() for df in results]
    seed_mean_times = [df["time_seconds"].mean() for df in results if "time_seconds" in df.columns]
    if not seed_mean_times:
        seed_mean_times = [0.0] * len(results)

    # Try loading saved config
    config = {}
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    return {
        "name": exp_dir.name,
        "path": exp_dir,
        "config": config,
        "seeds": seeds,
        "results": results,
        "step_data": step_data,
        "n_proteins": len(results[0]),
        "mean_lddt": float(np.mean(seed_mean_lddts)),
        "std_lddt": float(np.std(seed_mean_lddts)),
        "mean_time": float(np.mean(seed_mean_times)),
        "std_time": float(np.std(seed_mean_times)),
        "seed_lddts": seed_mean_lddts,
        "seed_times": seed_mean_times,
    }


def _flatten_config(cfg, prefix=""):
    """Flatten a nested config dict into dot-separated key-value pairs."""
    flat = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        else:
            flat[key] = v
    return flat


# Keys that are always different / not informative for comparison
_IGNORE_KEYS = {
    "seed", "df_path", "input.pdb_dir", "input.msa_dir", "input.summary_file",
    "columns.id_column", "columns.sequence_column", "columns.chain_id_column",
    "use_true_pdb", "use_chain_name", "renumber_pdb",
}

# Compact key aliases for readability in labels / plots
_KEY_ALIAS = {
    "lr": "lr",
    "ags": "ags",
    "lora_rank": "rank",
    "lora_alpha": "alpha",
    "gradient_clip": "clip",
    "gradient_clip_max_norm": "clip_norm",
    "msa_sampling_strategy": "msa_strat",
    "confidence_collapse_ratio": "cc_ratio",
    "confidence_collapse_patience": "cc_pat",
    "msa": "msa",
    "steps": "steps",
}


def make_diff_labels(experiments):
    """Compare configs across all experiments and build labels from changed params only.

    Returns a list of label strings, one per experiment.
    """
    configs = [exp.get("config", {}) for exp in experiments]
    if not any(configs):
        return [exp["name"] for exp in experiments]

    # Flatten all configs
    flat_configs = [_flatten_config(cfg) for cfg in configs]

    # Collect all keys
    all_keys = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())

    # Remove ignored keys
    all_keys -= _IGNORE_KEYS

    # Find keys that differ across experiments
    diff_keys = []
    for key in sorted(all_keys):
        values = [fc.get(key, "__MISSING__") for fc in flat_configs]
        if len(set(str(v) for v in values)) > 1:
            diff_keys.append(key)

    if not diff_keys:
        # Nothing differs – use short directory names
        return [exp["name"] for exp in experiments]

    # Build labels from differing keys only
    labels = []
    for i, exp in enumerate(experiments):
        fc = flat_configs[i]
        parts = []
        for key in diff_keys:
            val = fc.get(key, "—")
            # Short key name: use alias or last component after dot
            raw_key = key.split(".")[-1]
            short_key = _KEY_ALIAS.get(raw_key, raw_key)
            # Format value nicely
            if isinstance(val, float):
                parts.append(f"{short_key}={val:g}")
            elif isinstance(val, bool):
                parts.append(f"{short_key}={'T' if val else 'F'}")
            else:
                parts.append(f"{short_key}={val}")
        labels.append(", ".join(parts) if parts else exp["name"])

    # Handle duplicate labels (same config, different runs) by appending run index
    seen = {}
    for i, lbl in enumerate(labels):
        if lbl in seen:
            seen[lbl] += 1
            labels[i] = f"{lbl} (run {seen[lbl]})"
        else:
            seen[lbl] = 1
    # Go back and mark the first occurrence too if it has duplicates
    first_occ = {}
    for i, lbl in enumerate(labels):
        base = lbl.split(" (run ")[0]
        if base in seen and seen[base] > 1 and " (run " not in lbl:
            labels[i] = f"{lbl} (run 1)"

    return labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_comparison_plots(experiments, output_dir):
    """Generate comparison plots across experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(experiments)
    labels = make_diff_labels(experiments)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))

    point_color = "#4CAF50"
    rng = np.random.default_rng(42)

    # ==================================================================
    # Plot 1: Bar chart – Mean LDDT per experiment (with seed scatter)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(max(6, 2 + 1.5 * n), 6))
    x = np.arange(n)
    lddts = [e["mean_lddt"] for e in experiments]
    lddt_errs = [e["std_lddt"] for e in experiments]

    ax.bar(x, lddts, color=colors[:n], alpha=0.35, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, lddts, yerr=lddt_errs, fmt="none", ecolor="black", capsize=6, linewidth=2)

    # Scatter per-seed points
    for i, exp in enumerate(experiments):
        pts = np.array(exp["seed_lddts"])
        jitter = rng.uniform(-0.15, 0.15, len(pts))
        ax.scatter(np.full(len(pts), i) + jitter, pts,
                   color=point_color, edgecolors="black", linewidths=0.5,
                   s=60, zorder=5)

    for i, (val, err) in enumerate(zip(lddts, lddt_errs)):
        ax.text(i, val + err + 0.005,
                f"{val:.4f}±{err:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Mean LDDT", fontsize=12)
    ax.set_title("Mean LDDT Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_lddt.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==================================================================
    # Plot 2: Bar chart – Mean Time per experiment (with seed scatter)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(max(6, 2 + 1.5 * n), 6))
    times = [e["mean_time"] for e in experiments]
    time_errs = [e["std_time"] for e in experiments]

    ax.bar(x, times, color=colors[:n], alpha=0.35, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, times, yerr=time_errs, fmt="none", ecolor="black", capsize=6, linewidth=2)

    for i, exp in enumerate(experiments):
        pts = np.array(exp["seed_times"])
        jitter = rng.uniform(-0.15, 0.15, len(pts))
        ax.scatter(np.full(len(pts), i) + jitter, pts,
                   color=point_color, edgecolors="black", linewidths=0.5,
                   s=60, zorder=5)

    for i, (val, err) in enumerate(zip(times, time_errs)):
        ax.text(i, val + err + 0.5,
                f"{val:.1f}±{err:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Mean Time per Protein (s)", fontsize=12)
    ax.set_title("Mean Time Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==================================================================
    # Plot 3: Grouped bar – LDDT + Time side by side (dual y-axis, seed scatter)
    # ==================================================================
    fig, ax1 = plt.subplots(figsize=(max(7, 2 + 2.0 * n), 6))
    width = 0.35
    x_lddt = x - width / 2
    x_time = x + width / 2

    ax1.bar(x_lddt, lddts, width, color="#2196F3", alpha=0.35,
            edgecolor="black", linewidth=0.5, label="Mean LDDT")
    ax1.errorbar(x_lddt, lddts, yerr=lddt_errs, fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    for i, exp in enumerate(experiments):
        pts = np.array(exp["seed_lddts"])
        jitter = rng.uniform(-0.06, 0.06, len(pts))
        ax1.scatter(np.full(len(pts), x_lddt[i]) + jitter, pts,
                    color=point_color, edgecolors="black", linewidths=0.5,
                    s=50, zorder=5)

    ax1.set_ylabel("Mean LDDT", fontsize=12, color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")

    ax2 = ax1.twinx()
    ax2.bar(x_time, times, width, color="#FF9800", alpha=0.35,
            edgecolor="black", linewidth=0.5, label="Mean Time (s)")
    ax2.errorbar(x_time, times, yerr=time_errs, fmt="none", ecolor="black", capsize=5, linewidth=1.5)

    for i, exp in enumerate(experiments):
        pts = np.array(exp["seed_times"])
        jitter = rng.uniform(-0.06, 0.06, len(pts))
        ax2.scatter(np.full(len(pts), x_time[i]) + jitter, pts,
                    color=point_color, edgecolors="black", linewidths=0.5,
                    s=50, zorder=5)

    ax2.set_ylabel("Mean Time per Protein (s)", fontsize=12, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax1.set_title("Benchmark Comparison: LDDT & Time", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_combined.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==================================================================
    # Plot 4: Mean LDDT per step – all experiments overlaid
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    has_step_data = False
    for i, exp in enumerate(experiments):
        if not exp["step_data"]:
            continue
        # Average across seeds first
        seed_means = []
        for seed, sdf in exp["step_data"].items():
            if "lddt" in sdf.columns and not sdf["lddt"].dropna().empty:
                seed_means.append(sdf.groupby("step")["lddt"].mean())
        if not seed_means:
            continue
        has_step_data = True
        common_steps = seed_means[0].index
        for sm in seed_means[1:]:
            common_steps = common_steps.intersection(sm.index)
        aligned = np.array([[sm[s] for s in common_steps] for sm in seed_means])
        mean_vals = aligned.mean(axis=0)
        std_vals = aligned.std(axis=0)
        ax.plot(common_steps, mean_vals, "o-", color=colors[i], linewidth=2,
                markersize=4, label=labels[i])
        ax.fill_between(common_steps, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.15, color=colors[i])

    if has_step_data:
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Mean LDDT", fontsize=12)
        ax.set_title("Mean LDDT per Step – All Experiments", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_lddt_per_step.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==================================================================
    # Plot 5: Summary table as image
    # ==================================================================
    table_data = []
    for i, exp in enumerate(experiments):
        table_data.append([
            labels[i],
            f"{exp['mean_lddt']:.4f} ± {exp['std_lddt']:.4f}",
            f"{exp['mean_time']:.1f} ± {exp['std_time']:.1f}",
            str(exp["n_proteins"]),
            str(len(exp["seeds"])),
        ])
    col_labels = ["Experiment", "Mean LDDT", "Mean Time (s)", "Proteins", "Seeds"]

    fig, ax = plt.subplots(figsize=(max(10, 2 + 2 * n), 1 + 0.5 * n))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#e0e0e0"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Highlight best LDDT row
    best_idx = int(np.argmax([e["mean_lddt"] for e in experiments]))
    for col in range(len(col_labels)):
        table[best_idx + 1, col].set_facecolor("#c8e6c9")

    ax.set_title("Benchmark Summary", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Also save as CSV
    summary_df = pd.DataFrame(table_data, columns=col_labels)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    logging.info(f"Comparison plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Parameter sweep heatmap
# ---------------------------------------------------------------------------

def _find_single_param_sweeps(experiments, flat_configs, varying_keys):
    """Find groups of experiments that differ in exactly one parameter.

    Returns list of dicts:
        {param, param_short, context_label, points: [(val_str, lddt, std, exp_idx)]}
    """
    all_keys = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())
    all_keys -= _IGNORE_KEYS

    sweeps = []

    for param in varying_keys:
        # Context = all keys except this param
        other_keys = sorted(all_keys - {param})

        groups = {}
        for idx, fc in enumerate(flat_configs):
            ctx = tuple((k, str(fc.get(k, ""))) for k in other_keys)
            groups.setdefault(ctx, []).append(idx)

        for ctx, indices in groups.items():
            if len(indices) < 2:
                continue

            # Sort by param value
            def _sort_key(i, _p=param):
                v = flat_configs[i].get(_p, 0)
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return str(v)

            indices.sort(key=_sort_key)

            # Context label: only show *other* varying params
            ctx_parts = []
            for k in varying_keys:
                if k == param:
                    continue
                val = flat_configs[indices[0]].get(k, "—")
                sk = _KEY_ALIAS.get(k.split(".")[-1], k.split(".")[-1])
                ctx_parts.append(f"{sk}={val:g}" if isinstance(val, float) else f"{sk}={val}")
            ctx_label = ", ".join(ctx_parts) if ctx_parts else "default"

            points = []
            for i in indices:
                val = flat_configs[i].get(param, "?")
                val_str = f"{val:g}" if isinstance(val, float) else str(val)
                points.append((val_str, experiments[i]["mean_lddt"],
                               experiments[i]["std_lddt"], i))

            short_param = _KEY_ALIAS.get(param.split(".")[-1], param.split(".")[-1])
            sweeps.append(dict(param=param, param_short=short_param,
                               context_label=ctx_label, points=points))

    return sweeps


def _find_pair_param_sweeps(experiments, flat_configs, varying_keys, single_sweeps):
    """Find groups where exactly two params co-vary (e.g. rank+alpha).

    Only returns groups not already found by single-param sweeps.
    """
    all_keys = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())
    all_keys -= _IGNORE_KEYS

    # Index sets already explained by single sweeps
    single_index_sets = set()
    for sw in single_sweeps:
        single_index_sets.add(frozenset(p[3] for p in sw["points"]))

    sweeps = []

    for ip, p1 in enumerate(varying_keys):
        for p2 in varying_keys[ip + 1:]:
            other_keys = sorted(all_keys - {p1, p2})

            groups = {}
            for idx, fc in enumerate(flat_configs):
                ctx = tuple((k, str(fc.get(k, ""))) for k in other_keys)
                groups.setdefault(ctx, []).append(idx)

            for ctx, indices in groups.items():
                if len(indices) < 2:
                    continue
                if frozenset(indices) in single_index_sets:
                    continue

                def _sort_key(i, _p=p1):
                    v = flat_configs[i].get(_p, 0)
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return str(v)

                indices.sort(key=_sort_key)

                ctx_parts = []
                for k in varying_keys:
                    if k in (p1, p2):
                        continue
                    val = flat_configs[indices[0]].get(k, "—")
                    sk = _KEY_ALIAS.get(k.split(".")[-1], k.split(".")[-1])
                    ctx_parts.append(f"{sk}={val:g}" if isinstance(val, float) else f"{sk}={val}")
                ctx_label = ", ".join(ctx_parts) if ctx_parts else "default"

                points = []
                for i in indices:
                    v1 = flat_configs[i].get(p1, "?")
                    v2 = flat_configs[i].get(p2, "?")
                    v1s = f"{v1:g}" if isinstance(v1, float) else str(v1)
                    v2s = f"{v2:g}" if isinstance(v2, float) else str(v2)
                    points.append((f"{v1s}/{v2s}", experiments[i]["mean_lddt"],
                                   experiments[i]["std_lddt"], i))

                sp1 = _KEY_ALIAS.get(p1.split(".")[-1], p1.split(".")[-1])
                sp2 = _KEY_ALIAS.get(p2.split(".")[-1], p2.split(".")[-1])
                sweeps.append(dict(param=f"{p1}+{p2}", param_short=f"{sp1}+{sp2}",
                                   context_label=ctx_label, points=points))

    return sweeps


def _short_context_label(sweep, varying_keys):
    """Build a minimal context label – only params that distinguish contexts
    for the *same* parameter, not the full set of other varying keys."""
    return sweep["context_label"]


def _minimal_context_labels(sweep_list, varying_keys):
    """For a list of sweeps of the SAME parameter, find only the keys
    that actually differ between contexts and use those as labels."""
    if len(sweep_list) <= 1:
        return [sw["context_label"] for sw in sweep_list]

    # Parse the context labels back to extract which keys differ
    # across the sweep contexts
    # Each sweep's context_label is like "clip=True, clip_norm=1, alpha=256, ..."
    # We parse them to find which parts differ
    parsed = []
    for sw in sweep_list:
        parts = {}
        for token in sw["context_label"].split(", "):
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        parsed.append(parts)

    # Find keys that actually differ between contexts
    all_ctx_keys = set()
    for p in parsed:
        all_ctx_keys.update(p.keys())

    diff_ctx_keys = []
    for k in sorted(all_ctx_keys):
        vals = {p.get(k, "?") for p in parsed}
        if len(vals) > 1:
            diff_ctx_keys.append(k)

    if not diff_ctx_keys:
        return [f"#{i + 1}" for i in range(len(sweep_list))]

    labels = []
    for p in parsed:
        parts = [f"{k}={p.get(k, '?')}" for k in diff_ctx_keys]
        labels.append("  ".join(parts))
    return labels


def generate_parameter_sweep_plots(experiments, output_dir):
    """Heatmap-style plots: for each parameter, show how LDDT changes.

    Experiments can appear in multiple subplots if they participate in
    multiple single-parameter sweeps.
    """
    output_dir = Path(output_dir)

    configs = [exp.get("config", {}) for exp in experiments]
    flat_configs = [_flatten_config(cfg) for cfg in configs]

    all_keys = set()
    for fc in flat_configs:
        all_keys.update(fc.keys())
    all_keys -= _IGNORE_KEYS

    varying_keys = []
    for key in sorted(all_keys):
        values = [str(fc.get(key, "__MISSING__")) for fc in flat_configs]
        if len(set(values)) > 1:
            varying_keys.append(key)

    if not varying_keys:
        return

    single_sweeps = _find_single_param_sweeps(experiments, flat_configs, varying_keys)
    pair_sweeps = _find_pair_param_sweeps(experiments, flat_configs, varying_keys, single_sweeps)

    # Only keep pair sweeps that involve co-varying params (rank+alpha) —
    # drop noisy multi-param combos to keep the plot clean
    _CO_VARY_PAIRS = {"rank+alpha", "alpha+rank"}
    clean_pair_sweeps = [sw for sw in pair_sweeps
                         if sw["param_short"] in _CO_VARY_PAIRS]
    all_sweeps = single_sweeps + clean_pair_sweeps

    if not all_sweeps:
        return

    # Group sweeps by parameter
    param_order = []
    param_groups = {}
    for sw in all_sweeps:
        pname = sw["param_short"]
        if pname not in param_groups:
            param_order.append(pname)
            param_groups[pname] = []
        param_groups[pname].append(sw)

    # ---- Compute layout ----
    all_lddts = [e["mean_lddt"] for e in experiments]
    vmin = min(all_lddts) - 0.01
    vmax = max(all_lddts) + 0.01
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    n_params = len(param_order)

    # Height per subplot row, with padding
    total_ctx_rows = sum(len(param_groups[p]) for p in param_order)
    cell_h = 0.75  # height per context row in inches
    subplot_title_h = 0.7  # title + x-label space
    fig_h = max(4, total_ctx_rows * cell_h + n_params * subplot_title_h + 2)
    fig_w = 12

    fig, axes = plt.subplots(n_params, 1, figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.7})
    if n_params == 1:
        axes = [axes]

    for row, pname in enumerate(param_order):
        ax = axes[row]
        sweep_list = param_groups[pname]

        # Collect all unique x values for this parameter
        all_x = []
        for sw in sweep_list:
            for val, *_ in sw["points"]:
                if val not in all_x:
                    all_x.append(val)
        try:
            all_x.sort(key=float)
        except ValueError:
            all_x.sort()

        x_idx = {v: i for i, v in enumerate(all_x)}
        n_x = len(all_x)
        n_rows = len(sweep_list)

        # Minimal context labels — only show what differs between rows
        ctx_labels = _minimal_context_labels(sweep_list, varying_keys)

        # --- Draw cells ---
        sq_size = 0.88  # relative to grid spacing = 1
        for ctx_i, sw in enumerate(sweep_list):
            for val, lddt, std, exp_i in sw["points"]:
                xi = x_idx[val]
                color = cmap(norm(lddt))
                rect = plt.Rectangle(
                    (xi - sq_size / 2, ctx_i - sq_size / 2),
                    sq_size, sq_size,
                    facecolor=color, edgecolor="white", linewidth=2,
                    joinstyle="round",
                )
                ax.add_patch(rect)

                # Text contrast
                luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                txt_c = "white" if luma < 0.5 else "black"

                # Main LDDT value
                ax.text(xi, ctx_i - 0.05, f"{lddt:.3f}",
                        ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_c,
                        fontfamily="monospace")
                # ± std below
                ax.text(xi, ctx_i + 0.28, f"±{std:.3f}",
                        ha="center", va="center",
                        fontsize=8, color=txt_c, alpha=0.8,
                        fontfamily="monospace")

        # --- Axes formatting ---
        ax.set_xlim(-0.55, n_x - 0.45)
        ax.set_ylim(n_rows - 0.55, -0.55)  # invert y

        ax.set_xticks(range(n_x))
        ax.set_xticklabels(all_x, fontsize=12, fontweight="bold")
        ax.set_xlabel(pname, fontsize=13, fontweight="bold", labelpad=6)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ctx_labels, fontsize=10)

        ax.set_title(f"Effect of  {pname}", fontsize=15, fontweight="bold", pad=10)

        # Remove chart border, add light grid
        ax.set_frame_on(False)
        ax.tick_params(length=0)  # hide tick marks

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label="Mean LDDT", shrink=0.5,
                        pad=0.03, aspect=30)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Mean LDDT", fontsize=12, fontweight="bold")

    plt.savefig(output_dir / "parameter_sweeps.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Parameter sweep heatmap saved to {output_dir / 'parameter_sweeps.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple ProteinTTT benchmark experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment_dirs", nargs="*",
                        help="Paths to individual experiment directories")
    parser.add_argument("--benchmark_dir", type=str, default=None,
                        help="Root benchmark directory – auto-discovers all experiments inside")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save comparison plots (default: <benchmark_dir>/comparison)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Collect experiment directories
    exp_dirs = []
    if args.experiment_dirs:
        exp_dirs = [Path(d) for d in args.experiment_dirs]
    elif args.benchmark_dir:
        bdir = Path(args.benchmark_dir)
        # Each sub-directory that contains seed_* is an experiment
        for child in sorted(bdir.iterdir()):
            if child.is_dir() and any(child.glob("seed_*")):
                exp_dirs.append(child)
    else:
        parser.error("Provide experiment directories or --benchmark_dir")

    if not exp_dirs:
        logging.error("No experiment directories found.")
        sys.exit(1)

    logging.info(f"Found {len(exp_dirs)} experiment(s)")

    # Load data
    experiments = []
    for d in exp_dirs:
        exp = load_experiment(d)
        if exp is None:
            logging.warning(f"Skipping {d} – no valid results")
            continue
        logging.info(f"  {exp['name']}: LDDT={exp['mean_lddt']:.4f}±{exp['std_lddt']:.4f}, "
                     f"Time={exp['mean_time']:.1f}±{exp['std_time']:.1f}s, "
                     f"seeds={exp['seeds']}, proteins={exp['n_proteins']}")
        experiments.append(exp)

    if not experiments:
        logging.error("No valid experiments to compare.")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.benchmark_dir:
        output_dir = Path(args.benchmark_dir) / "comparison"
    else:
        output_dir = exp_dirs[0].parent / "comparison"

    generate_comparison_plots(experiments, output_dir)
    generate_parameter_sweep_plots(experiments, output_dir)
    logging.info("Done!")


if __name__ == "__main__":
    main()
