"""
Visualize a ProteinTTT optimization log TSV file.

Produces four panels:
  1. Line chart — mean pLDDT and TM-score across optimization steps (T1)
  2. Heatmap — per-residue pLDDT across steps (T2)
  3. Sparklines — per-residue pLDDT trajectories alongside heatmap (T2/T6)
  4. Small multiples — pLDDT profiles at selected steps (T3)

Usage:
    python scripts/visualize_log.py data/cameo_testset/logs/7eqs_A_log.tsv
"""

import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# 1. Parse the TSV
# ---------------------------------------------------------------------------

def parse_log(path):
    """Return step-level metrics and a (steps x residues) pLDDT matrix."""
    with open(path) as f:
        lines = f.readlines()

    # Header is line 0
    header = lines[0].strip().split("\t")

    # Identify step-rows: lines with >= 10 tab-separated fields
    step_rows = []
    step_line_indices = []
    for i, line in enumerate(lines[1:], start=1):
        if len(line.split("\t")) >= 10:
            step_rows.append(line.strip().split("\t"))
            step_line_indices.append(i)

    # Extract step-level scalars
    steps = [int(r[0]) for r in step_rows]
    plddt_mean = [float(r[8]) for r in step_rows]
    tm_scores = [float(r[9]) for r in step_rows]

    # Extract per-residue pLDDT from CA B-factors in embedded PDB blocks
    n_steps = len(step_line_indices)
    # Determine block boundaries
    boundaries = step_line_indices + [len(lines)]

    plddt_matrix = []
    for idx in range(n_steps):
        start = boundaries[idx] + 1  # line after the step row
        end = boundaries[idx + 1]
        residue_plddts = []
        for j in range(start, end):
            line = lines[j]
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                bfactor = float(line[60:66])
                residue_plddts.append(bfactor)
            elif line.startswith("END"):
                break
        plddt_matrix.append(residue_plddts)

    plddt_matrix = np.array(plddt_matrix)  # shape: (n_steps, n_residues)
    return steps, plddt_mean, tm_scores, plddt_matrix


# ---------------------------------------------------------------------------
# 2. AlphaFold-style colormap (blue > cyan > yellow > orange)
# ---------------------------------------------------------------------------

AF_CMAP = LinearSegmentedColormap.from_list(
    "alphafold",
    [
        (0.0, "#FF7D45"),   # very low  (orange)
        (0.5, "#FFDB13"),   # low       (yellow)
        (0.7, "#65CBF3"),   # confident (cyan)
        (1.0, "#0053D6"),   # very high (blue)
    ],
)

# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_all(steps, plddt_mean, tm_scores, plddt_matrix, out_prefix):
    n_steps, n_res = plddt_matrix.shape
    residue_ids = np.arange(1, n_res + 1)

    # ---- Figure 1: Line chart + Heatmap + Sparklines ----------------------
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[5, 1],
        height_ratios=[1, 2],
        hspace=0.25,
        wspace=0.05,
    )

    # -- Panel A: Mean pLDDT & TM-score line chart -------------------------
    ax_line = fig.add_subplot(gs[0, 0])
    color_plddt = "#0053D6"
    color_tm = "#FF7D45"

    ax_line.plot(steps, plddt_mean, "o-", color=color_plddt, label="Mean pLDDT", markersize=4)
    ax_line.set_xlabel("Optimization step")
    ax_line.set_ylabel("Mean pLDDT", color=color_plddt)
    ax_line.tick_params(axis="y", labelcolor=color_plddt)

    ax_tm = ax_line.twinx()
    ax_tm.plot(steps, tm_scores, "s-", color=color_tm, label="TM-score", markersize=4)
    ax_tm.set_ylabel("TM-score", color=color_tm)
    ax_tm.tick_params(axis="y", labelcolor=color_tm)

    lines1, labels1 = ax_line.get_legend_handles_labels()
    lines2, labels2 = ax_tm.get_legend_handles_labels()
    ax_line.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax_line.set_title("Global optimization progress")

    # -- Panel B: Heatmap ---------------------------------------------------
    ax_heat = fig.add_subplot(gs[1, 0])
    vmin, vmax = 0, 100
    im = ax_heat.imshow(
        plddt_matrix,
        aspect="auto",
        cmap=AF_CMAP,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[0.5, n_res + 0.5, -0.5, n_steps - 0.5],
    )
    ax_heat.set_xlabel("Residue index")
    ax_heat.set_ylabel("Optimization step")
    ax_heat.set_title("Per-residue pLDDT across optimization steps")
    plt.colorbar(im, ax=ax_heat, label="pLDDT", shrink=0.8)

    # -- Panel C: Sparklines (subset of residues) ---------------------------
    # Select up to 40 evenly spaced residues for readability
    max_spark = 40
    if n_res > max_spark:
        spark_idx = np.linspace(0, n_res - 1, max_spark, dtype=int)
    else:
        spark_idx = np.arange(n_res)

    ax_spark = fig.add_subplot(gs[1, 1])
    ax_spark.set_title("Sparklines", fontsize=10)
    ax_spark.set_xlim(0, n_steps - 1)
    ax_spark.set_ylim(-0.5, len(spark_idx) - 0.5)
    ax_spark.set_xlabel("Step")
    ax_spark.set_yticks(range(len(spark_idx)))
    ax_spark.set_yticklabels([str(residue_ids[i]) for i in spark_idx], fontsize=6)
    ax_spark.set_ylabel("Residue")

    for row, res_i in enumerate(spark_idx):
        vals = plddt_matrix[:, res_i]
        # Normalize to [0,1] within the row for visual clarity
        lo, hi = vals.min(), vals.max()
        if hi - lo < 1e-3:
            normed = np.full_like(vals, 0.5)
        else:
            normed = (vals - lo) / (hi - lo)
        # Scale to fit within the row band (height ~0.8)
        y = row - 0.4 + 0.8 * normed
        ax_spark.plot(np.arange(n_steps), y, linewidth=0.7, color="black")

    ax_spark.tick_params(axis="both", which="both", length=2)

    fig.suptitle("ProteinTTT Optimization — Heatmap & Sparklines", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_heatmap_sparklines.png", dpi=200, bbox_inches="tight")
    print(f"Saved {out_prefix}_heatmap_sparklines.png")

    # ---- Figure 2: Small multiples ----------------------------------------
    # Pick steps: first, ~1/4, ~1/2, ~3/4, peak mean-pLDDT, last
    peak_step = int(np.argmax(plddt_mean))
    chosen = sorted(set([0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, peak_step, n_steps - 1]))

    n_panels = len(chosen)
    fig2, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, si in zip(axes, chosen):
        vals = plddt_matrix[si]
        # Colour by AlphaFold bands
        colors = []
        for v in vals:
            if v >= 90:
                colors.append("#0053D6")
            elif v >= 70:
                colors.append("#65CBF3")
            elif v >= 50:
                colors.append("#FFDB13")
            else:
                colors.append("#FF7D45")
        ax.bar(residue_ids, vals, color=colors, width=1.0, edgecolor="none")
        ax.set_ylabel("pLDDT")
        ax.set_ylim(0, 100)
        ax.set_title(f"Step {steps[si]}  (mean pLDDT = {plddt_mean[si]:.1f}, TM = {tm_scores[si]:.3f})", fontsize=10)
        # Band guidelines
        for y in [50, 70, 90]:
            ax.axhline(y, color="grey", linewidth=0.4, linestyle="--")

    axes[-1].set_xlabel("Residue index")
    fig2.suptitle("ProteinTTT — Per-residue pLDDT at selected steps (Small Multiples)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_small_multiples.png", dpi=200, bbox_inches="tight")
    print(f"Saved {out_prefix}_small_multiples.png")

    plt.close("all")

    # ---- Figure 3: Parallel coordinates -----------------------------------
    # Each vertical axis = one residue, each polyline = one step's pLDDT profile.
    # To keep it readable: subset of residues + selected steps only.

    # Select residues: every 8th residue (≈38 axes for 304 residues)
    res_stride = max(1, n_res // 40)
    res_idx = np.arange(0, n_res, res_stride)
    res_labels = residue_ids[res_idx]

    # Select steps: initial, 1/4, 1/2, peak, 3/4, final
    peak_step_idx = int(np.argmax(plddt_mean))
    sel_steps = sorted(set([
        0, n_steps // 4, n_steps // 2, peak_step_idx, 3 * n_steps // 4, n_steps - 1
    ]))

    # Colormap for step lines: viridis from early (purple) to late (yellow)
    step_cmap = plt.cm.viridis
    step_norm = plt.Normalize(vmin=0, vmax=n_steps - 1)

    fig3, ax_pc = plt.subplots(figsize=(max(14, len(res_idx) * 0.4), 6))

    x_positions = np.arange(len(res_idx))

    for si in sel_steps:
        vals = plddt_matrix[si, res_idx]
        color = step_cmap(step_norm(si))
        lw = 2.5 if si == peak_step_idx else 1.4
        alpha = 1.0 if si == peak_step_idx else 0.7
        label = f"Step {steps[si]} (pLDDT={plddt_mean[si]:.1f})"
        if si == peak_step_idx:
            label += " ★ peak"
        ax_pc.plot(x_positions, vals, color=color, linewidth=lw, alpha=alpha, label=label)

    # AlphaFold confidence band shading
    ax_pc.axhspan(0, 50, color="#FF7D45", alpha=0.08)
    ax_pc.axhspan(50, 70, color="#FFDB13", alpha=0.08)
    ax_pc.axhspan(70, 90, color="#65CBF3", alpha=0.08)
    ax_pc.axhspan(90, 100, color="#0053D6", alpha=0.08)
    for y in [50, 70, 90]:
        ax_pc.axhline(y, color="grey", linewidth=0.4, linestyle="--")

    # Vertical grid lines for each residue axis
    for xp in x_positions:
        ax_pc.axvline(xp, color="lightgrey", linewidth=0.3, zorder=0)

    ax_pc.set_xticks(x_positions)
    ax_pc.set_xticklabels(res_labels, fontsize=7, rotation=90)
    ax_pc.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax_pc.set_ylim(0, 100)
    ax_pc.set_xlabel("Residue index")
    ax_pc.set_ylabel("pLDDT")
    ax_pc.set_title("Parallel Coordinates — pLDDT profile at selected optimization steps")
    ax_pc.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig3.tight_layout()
    fig3.savefig(f"{out_prefix}_parallel_coords.png", dpi=200, bbox_inches="tight")
    print(f"Saved {out_prefix}_parallel_coords.png")

    plt.close("all")

    # ---- Figure 4: Flexible Linked Axes (connected SPLOM) -----------------
    # Scatterplot matrix of pLDDT at selected steps, with lines connecting
    # the same residue across adjacent panels — hybrid of SPLOM and
    # parallel coordinates.

    # Select steps for axes
    peak_si = int(np.argmax(plddt_mean))
    splom_steps = sorted(set([0, peak_si, n_steps - 1]))
    n_axes = len(splom_steps)
    step_labels = [f"Step {steps[s]}" for s in splom_steps]

    # Per-residue data columns: shape (n_res, n_axes)
    cols = np.column_stack([plddt_matrix[s] for s in splom_steps])

    # Colour each residue by its mean pLDDT across selected steps
    res_mean = cols.mean(axis=1)
    res_norm = plt.Normalize(vmin=0, vmax=100)
    res_cmap = AF_CMAP

    fig4, axes_grid = plt.subplots(
        n_axes, n_axes,
        figsize=(4 * n_axes + 1, 4 * n_axes + 1),
        squeeze=False,
    )

    pad = 5  # axis padding in pLDDT units
    lo, hi = max(0, cols.min() - pad), min(100, cols.max() + pad)

    for row in range(n_axes):
        for col in range(n_axes):
            ax = axes_grid[row][col]

            if row == col:
                # Diagonal: histogram of pLDDT at this step
                ax.hist(cols[:, col], bins=30, color="steelblue", edgecolor="white",
                        alpha=0.8, range=(0, 100))
                ax.set_xlim(lo, hi)
                ax.set_ylabel("Count")
            else:
                # Off-diagonal: scatter of step[col] vs step[row]
                colors = res_cmap(res_norm(res_mean))
                ax.scatter(cols[:, col], cols[:, row], c=colors, s=12,
                           edgecolors="none", alpha=0.85, zorder=3)

                # Lines linking same residue between adjacent panels
                # Draw lines only for horizontally or vertically adjacent pairs
                if abs(row - col) == 1:
                    for r in range(len(cols)):
                        ax.plot(
                            [cols[r, col]], [cols[r, row]],
                            color=res_cmap(res_norm(res_mean[r])),
                            alpha=0.15, linewidth=0.5, zorder=1,
                        )

                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                # Identity line
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.3)

            # Axis labels
            if row == n_axes - 1:
                ax.set_xlabel(step_labels[col], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col == 0:
                if row != col:
                    ax.set_ylabel(step_labels[row], fontsize=10)
            else:
                ax.set_yticklabels([])

    # Add linking lines BETWEEN adjacent scatter panels (the key feature)
    # Draw on the figure using figure-level coordinates
    fig4.canvas.draw()

    for row in range(n_axes):
        for col in range(n_axes - 1):
            if row == col or row == col + 1:
                continue
            ax_left = axes_grid[row][col]
            ax_right = axes_grid[row][col + 1]

            for r in range(0, len(cols), 3):  # every 3rd residue to reduce clutter
                # Point in left panel: x=cols[r, col], y=cols[r, row]
                # Point in right panel: x=cols[r, col+1], y=cols[r, row]
                xy_left = ax_left.transData.transform((cols[r, col], cols[r, row]))
                xy_right = ax_right.transData.transform((cols[r, col + 1], cols[r, row]))

                # Convert to figure coords
                xy_left_fig = fig4.transFigure.inverted().transform(xy_left)
                xy_right_fig = fig4.transFigure.inverted().transform(xy_right)

                color = res_cmap(res_norm(res_mean[r]))
                line = plt.Line2D(
                    [xy_left_fig[0], xy_right_fig[0]],
                    [xy_left_fig[1], xy_right_fig[1]],
                    transform=fig4.transFigure,
                    color=color, alpha=0.2, linewidth=0.5,
                    clip_on=False,
                )
                fig4.lines.append(line)

    # Also add cross-panel links for vertically adjacent panels
    for col in range(n_axes):
        for row in range(n_axes - 1):
            if row == col or row + 1 == col:
                continue
            ax_top = axes_grid[row][col]
            ax_bot = axes_grid[row + 1][col]

            for r in range(0, len(cols), 3):
                xy_top = ax_top.transData.transform((cols[r, col], cols[r, row]))
                xy_bot = ax_bot.transData.transform((cols[r, col], cols[r, row + 1]))

                xy_top_fig = fig4.transFigure.inverted().transform(xy_top)
                xy_bot_fig = fig4.transFigure.inverted().transform(xy_bot)

                color = res_cmap(res_norm(res_mean[r]))
                line = plt.Line2D(
                    [xy_top_fig[0], xy_bot_fig[0]],
                    [xy_top_fig[1], xy_bot_fig[1]],
                    transform=fig4.transFigure,
                    color=color, alpha=0.2, linewidth=0.5,
                    clip_on=False,
                )
                fig4.lines.append(line)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=res_cmap, norm=res_norm)
    sm.set_array([])
    cbar = fig4.colorbar(sm, ax=axes_grid.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Mean pLDDT (across selected steps)")

    fig4.suptitle(
        "Flexible Linked Axes — Per-residue pLDDT across optimization steps",
        fontsize=14, y=1.01,
    )
    fig4.savefig(f"{out_prefix}_linked_splom.png", dpi=200, bbox_inches="tight")
    print(f"Saved {out_prefix}_linked_splom.png")

    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_combined_linked_splom(all_data, out_path):
    """
    Flexible Linked Axes plot combining all proteins.

    all_data: list of (name, steps, plddt_mean, tm_scores, plddt_matrix) tuples
    Each protein's residues are plotted, colored by protein identity.
    Axes = initial step, peak step, final step.
    """
    # Use a distinct colour per protein
    n_proteins = len(all_data)
    protein_cmap = plt.cm.tab20 if n_proteins <= 20 else plt.cm.gist_ncar

    # Collect per-protein columns: for each protein pick step 0, peak, final
    protein_cols_list = []   # each entry: (n_res, 3) array
    protein_names = []
    protein_colors = []

    for i, (name, steps, plddt_mean, tm_scores, plddt_matrix) in enumerate(all_data):
        n_steps = len(steps)
        peak_si = int(np.argmax(plddt_mean))
        sel = [0, peak_si, n_steps - 1]
        cols = np.column_stack([plddt_matrix[s] for s in sel])
        protein_cols_list.append(cols)
        protein_names.append(name)
        protein_colors.append(protein_cmap(i / max(n_proteins - 1, 1)))

    axis_labels = ["Initial (step 0)", "Peak step", "Final step"]
    n_axes = 3

    fig, axes_grid = plt.subplots(
        n_axes, n_axes,
        figsize=(4 * n_axes + 2, 4 * n_axes + 1),
        squeeze=False,
    )

    lo, hi = 0, 100

    for row in range(n_axes):
        for col in range(n_axes):
            ax = axes_grid[row][col]

            if row == col:
                # Diagonal: overlaid histograms per protein
                for pi, cols in enumerate(protein_cols_list):
                    ax.hist(cols[:, col], bins=30, range=(0, 100),
                            color=protein_colors[pi], alpha=0.4,
                            edgecolor="none", label=protein_names[pi])
                ax.set_xlim(lo, hi)
                ax.set_ylabel("Count")
            else:
                # Off-diagonal: scatter
                for pi, cols in enumerate(protein_cols_list):
                    ax.scatter(cols[:, col], cols[:, row],
                               c=[protein_colors[pi]], s=6,
                               edgecolors="none", alpha=0.5, zorder=3)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.3)

            if row == n_axes - 1:
                ax.set_xlabel(axis_labels[col], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(axis_labels[row], fontsize=10)
            elif col != 0:
                ax.set_yticklabels([])

    # Draw cross-panel linking lines between horizontally adjacent panels
    fig.canvas.draw()

    for row in range(n_axes):
        for col in range(n_axes - 1):
            if row == col or row == col + 1:
                continue
            ax_left = axes_grid[row][col]
            ax_right = axes_grid[row][col + 1]

            for pi, cols in enumerate(protein_cols_list):
                stride = max(1, len(cols) // 30)  # subsample for clarity
                for r in range(0, len(cols), stride):
                    xy_l = ax_left.transData.transform((cols[r, col], cols[r, row]))
                    xy_r = ax_right.transData.transform((cols[r, col + 1], cols[r, row]))
                    xy_lf = fig.transFigure.inverted().transform(xy_l)
                    xy_rf = fig.transFigure.inverted().transform(xy_r)
                    line = plt.Line2D(
                        [xy_lf[0], xy_rf[0]], [xy_lf[1], xy_rf[1]],
                        transform=fig.transFigure,
                        color=protein_colors[pi], alpha=0.08, linewidth=0.4,
                        clip_on=False,
                    )
                    fig.lines.append(line)

    # Vertical adjacent links
    for col in range(n_axes):
        for row in range(n_axes - 1):
            if row == col or row + 1 == col:
                continue
            ax_top = axes_grid[row][col]
            ax_bot = axes_grid[row + 1][col]

            for pi, cols in enumerate(protein_cols_list):
                stride = max(1, len(cols) // 30)
                for r in range(0, len(cols), stride):
                    xy_t = ax_top.transData.transform((cols[r, col], cols[r, row]))
                    xy_b = ax_bot.transData.transform((cols[r, col], cols[r, row + 1]))
                    xy_tf = fig.transFigure.inverted().transform(xy_t)
                    xy_bf = fig.transFigure.inverted().transform(xy_b)
                    line = plt.Line2D(
                        [xy_tf[0], xy_bf[0]], [xy_tf[1], xy_bf[1]],
                        transform=fig.transFigure,
                        color=protein_colors[pi], alpha=0.08, linewidth=0.4,
                        clip_on=False,
                    )
                    fig.lines.append(line)

    # Legend
    from matplotlib.lines import Line2D as L2D
    handles = [L2D([0], [0], marker="o", color="w", markerfacecolor=protein_colors[i],
                    markersize=6, label=protein_names[i]) for i in range(n_proteins)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=7, title="Protein", title_fontsize=9)

    fig.suptitle(
        "Flexible Linked Axes — All proteins (per-residue pLDDT: initial vs peak vs final)",
        fontsize=13, y=1.01,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close("all")


def process_one(tsv_path, out_prefix=None):
    """Parse and plot a single log file."""
    if out_prefix is None:
        out_prefix = tsv_path.replace(".tsv", "")
    print(f"\nParsing {tsv_path} ...")
    steps, plddt_mean, tm_scores, plddt_matrix = parse_log(tsv_path)
    print(f"  {len(steps)} steps, {plddt_matrix.shape[1]} residues")
    plot_all(steps, plddt_mean, tm_scores, plddt_matrix, out_prefix)


if __name__ == "__main__":
    import glob as glob_mod

    if len(sys.argv) < 2:
        print("Usage: python visualize_log.py <log.tsv | directory> [output_prefix]")
        sys.exit(1)

    target = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else None

    # If target is a directory, process all TSV files and produce combined SPLOM
    if os.path.isdir(target):
        tsv_files = sorted(glob_mod.glob(os.path.join(target, "*_log.tsv")))
        print(f"Found {len(tsv_files)} log files in {target}")

        all_data = []
        for tsv_path in tsv_files:
            try:
                name = os.path.basename(tsv_path).replace("_log.tsv", "")
                steps, plddt_mean, tm_scores, plddt_matrix = parse_log(tsv_path)
                print(f"  {name}: {len(steps)} steps, {plddt_matrix.shape[1]} residues")
                all_data.append((name, steps, plddt_mean, tm_scores, plddt_matrix))
                # Per-protein plots
                process_one(tsv_path)
            except Exception as e:
                print(f"  ERROR processing {tsv_path}: {e}")

        # Combined Flexible Linked Axes for all proteins
        if all_data:
            combined_out = os.path.join(target, "all_proteins_linked_splom.png")
            plot_combined_linked_splom(all_data, combined_out)
    else:
        process_one(target, out_prefix)

    print("\nAll done.")
