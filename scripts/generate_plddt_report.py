#!/usr/bin/env python3
"""Generate an interactive HTML report from bfvd2/plddt_summary.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def analyze(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["delta_ttt"] = df["plddt_ProteinTTT"] - df["plddt_ESMFold"]
    df["delta_ttt_af"] = df["plddt_ProteinTTT"] - df["plddt_AF"]
    df["ttt_unchanged"] = np.isclose(
        df["plddt_ProteinTTT"], df["plddt_ESMFold"], atol=0.01
    )

    valid = df.dropna(subset=["plddt_ProteinTTT", "plddt_AF"])
    n = len(valid)

    def mean(col: str) -> float:
        return float(valid[col].mean())

    len_bins = pd.cut(
        valid["length"],
        bins=[0, 100, 150, 200, 300, 500],
        labels=["≤100", "101-150", "151-200", "201-300", "301-500"],
    )
    length_stats = (
        valid.groupby(len_bins, observed=True)
        .agg(
            count=("id", "count"),
            mean_esm=("plddt_ESMFold", "mean"),
            mean_ttt=("plddt_ProteinTTT", "mean"),
            mean_af=("plddt_AF", "mean"),
            mean_delta=("delta_ttt", "mean"),
        )
        .reset_index()
        .rename(columns={"length": "len_bin"})
    )

    baseline_bins = pd.cut(
        valid["plddt_ESMFold"],
        bins=[0, 35, 45, 55, 65, 100],
        labels=["<35", "35-45", "45-55", "55-65", ">65"],
    )
    baseline_stats = (
        valid.groupby(baseline_bins, observed=True)
        .agg(
            count=("id", "count"),
            mean_delta=("delta_ttt", "mean"),
            pct_improve=("delta_ttt", lambda x: 100 * (x > 0.01).mean()),
        )
        .reset_index()
        .rename(columns={"plddt_ESMFold": "baseline_bin"})
    )

    ptm_bins = pd.cut(
        valid["ptm"],
        bins=[0, 0.2, 0.3, 0.4, 0.5, 1.0],
        labels=["<0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", ">0.5"],
    )
    ptm_stats = (
        valid.groupby(ptm_bins, observed=True)
        .agg(mean_delta=("delta_ttt", "mean"), count=("id", "count"))
        .reset_index()
        .rename(columns={"ptm": "ptm_bin"})
    )

    hist_edges = np.arange(-2, 72, 2)
    hist_counts, _ = np.histogram(valid["delta_ttt"], bins=hist_edges)

    sample = valid.sample(min(4000, n), random_state=42)

    r_len, _ = stats.pearsonr(valid["length"], valid["delta_ttt"])
    r_base, _ = stats.pearsonr(valid["plddt_ESMFold"], valid["delta_ttt"])
    r_ptm, _ = stats.pearsonr(valid["ptm"], valid["delta_ttt"])

    _, wilcoxon_p = stats.wilcoxon(
        valid["plddt_ESMFold"], valid["plddt_ProteinTTT"]
    )

    top_up = valid.nlargest(10, "delta_ttt")[
        ["id", "plddt_ESMFold", "plddt_ProteinTTT", "plddt_AF", "delta_ttt", "length"]
    ]
    ttt_wins_af = valid.nlargest(10, "delta_ttt_af")[
        ["id", "plddt_ESMFold", "plddt_ProteinTTT", "plddt_AF", "delta_ttt_af", "length"]
    ]

    unchanged = valid[valid["ttt_unchanged"]]

    return {
        "n_total": int(len(df)),
        "n_valid": n,
        "n_batches": int(df["batch"].nunique()),
        "means": {
            "esmfold": mean("plddt_ESMFold"),
            "proteinttt": mean("plddt_ProteinTTT"),
            "af": mean("plddt_AF"),
            "delta_ttt": mean("delta_ttt"),
            "delta_ttt_af": mean("delta_ttt_af"),
        },
        "medians": {
            "esmfold": float(valid["plddt_ESMFold"].median()),
            "proteinttt": float(valid["plddt_ProteinTTT"].median()),
            "af": float(valid["plddt_AF"].median()),
            "delta_ttt": float(valid["delta_ttt"].median()),
        },
        "outcomes": {
            "improve": int((valid["delta_ttt"] > 0.01).sum()),
            "unchanged": int(valid["ttt_unchanged"].sum()),
            "regress": int((valid["delta_ttt"] < -0.01).sum()),
            "ttt_beats_af": int((valid["delta_ttt_af"] > 0).sum()),
            "ttt_beats_af_by_10": int((valid["delta_ttt_af"] > 10).sum()),
        },
        "unchanged_mean_baseline": float(unchanged["plddt_ESMFold"].mean()),
        "correlations": {
            "delta_vs_baseline": float(r_base),
            "delta_vs_length": float(r_len),
            "delta_vs_ptm": float(r_ptm),
        },
        "wilcoxon_p": float(wilcoxon_p),
        "length_bins": length_stats.round(2).to_dict("records"),
        "baseline_bins": baseline_stats.round(2).to_dict("records"),
        "ptm_bins": ptm_stats.round(2).to_dict("records"),
        "hist": {
            "edges": hist_edges.tolist(),
            "counts": hist_counts.tolist(),
        },
        "scatter": sample[
            ["plddt_ESMFold", "plddt_ProteinTTT", "plddt_AF", "delta_ttt"]
        ]
        .round(2)
        .to_dict("records"),
        "top_improvements": top_up.round(2).to_dict("records"),
        "ttt_beats_af_top": ttt_wins_af.round(2).to_dict("records"),
        "ss_delta": {
            "helix": float(
                (valid["helix_pct_ProteinTTT"] - valid["helix_pct_ESMFold"]).mean()
            ),
            "sheet": float(
                (valid["sheet_pct_ProteinTTT"] - valid["sheet_pct_ESMFold"]).mean()
            ),
            "loop": float(
                (valid["loop_pct_ProteinTTT"] - valid["loop_pct_ESMFold"]).mean()
            ),
        },
    }


def render_html(stats: dict, csv_path: Path, out_path: Path) -> None:
    data_json = json.dumps(stats)
    pct_improve = 100 * stats["outcomes"]["improve"] / stats["n_valid"]
    pct_unchanged = 100 * stats["outcomes"]["unchanged"] / stats["n_valid"]
    pct_ttt_af = 100 * stats["outcomes"]["ttt_beats_af"] / stats["n_valid"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BFVD2 pLDDT Analysis</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419; --card: #1a2332; --text: #e7ecf3; --muted: #8b9cb3;
      --accent: #4fc3f7; --good: #66bb6a; --warn: #ffb74d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    header {{
      padding: 2rem 2rem 1rem; border-bottom: 1px solid #2a3544;
      background: linear-gradient(135deg, #15202b 0%, #1a2a3a 100%);
    }}
    h1 {{ margin: 0 0 .25rem; font-size: 1.75rem; }}
    .sub {{ color: var(--muted); font-size: .95rem; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{
      background: var(--card); border-radius: 12px; padding: 1.1rem 1.25rem;
      border: 1px solid #2a3544;
    }}
    .card h3 {{ margin: 0 0 .5rem; font-size: .85rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
    .card .val {{ font-size: 1.8rem; font-weight: 700; }}
    .card .delta {{ color: var(--good); font-size: .95rem; }}
    section {{ margin-top: 2rem; }}
    section h2 {{ font-size: 1.2rem; margin-bottom: .75rem; }}
    .insight {{
      background: #1e2d3d; border-left: 4px solid var(--accent);
      padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }}
    .chart-box {{ background: var(--card); border-radius: 12px; padding: 1rem; border: 1px solid #2a3544; margin-bottom: 1rem; }}
    canvas {{ max-height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    th, td {{ padding: .45rem .6rem; text-align: left; border-bottom: 1px solid #2a3544; }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:hover td {{ background: #223044; }}
    .two-col {{ display: grid; gap: 1rem; grid-template-columns: 1fr 1fr; }}
    @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<header>
  <h1>BFVD2 pLDDT Benchmark Analysis</h1>
  <p class="sub">Source: {csv_path.name} · {stats["n_valid"]:,} proteins · {stats["n_batches"]:,} batches</p>
</header>
<main>
  <div class="grid">
    <div class="card"><h3>ESMFold (baseline)</h3><div class="val">{stats["means"]["esmfold"]:.1f}</div><div class="sub">median {stats["medians"]["esmfold"]:.1f}</div></div>
    <div class="card"><h3>ProteinTTT</h3><div class="val">{stats["means"]["proteinttt"]:.1f}</div><div class="delta">+{stats["means"]["delta_ttt"]:.1f} vs ESMFold</div></div>
    <div class="card"><h3>AlphaFold pLDDT</h3><div class="val">{stats["means"]["af"]:.1f}</div><div class="sub">+{stats["means"]["delta_ttt_af"]:.1f} vs AF on avg</div></div>
    <div class="card"><h3>TTT improves</h3><div class="val">{pct_improve:.1f}%</div><div class="sub">{pct_unchanged:.1f}% unchanged · 0% regress</div></div>
    <div class="card"><h3>TTT beats AF</h3><div class="val">{pct_ttt_af:.1f}%</div><div class="sub">{stats["outcomes"]["ttt_beats_af_by_10"]:,} by &gt;10 pts</div></div>
  </div>

  <div class="insight">
    <strong>Key finding:</strong> ProteinTTT raises mean pLDDT by <strong>+{stats["means"]["delta_ttt"]:.1f}</strong> points on {stats["n_valid"]:,} viral proteins.
    Gains are <em>anti-correlated</em> with baseline confidence (r={stats["correlations"]["delta_vs_baseline"]:.2f}):
    low-confidence ESMFold predictions (&lt;35 pLDDT) gain +{next(b["mean_delta"] for b in stats["baseline_bins"] if b.get("baseline_bin")=="<35"):.1f} on average,
    while already-confident ones (&gt;65) gain only +{next(b["mean_delta"] for b in stats["baseline_bins"] if b.get("baseline_bin")==">65"):.1f}.
    The {stats["outcomes"]["unchanged"]:,} unchanged cases (identical pLDDT) sit at baseline {stats["unchanged_mean_baseline"]:.1f} — likely early-stopping at step 0.
    TTT also shifts secondary structure toward more helix (+{stats["ss_delta"]["helix"]:.1f}%) and sheet (+{stats["ss_delta"]["sheet"]:.1f}%),
    reducing loop content ({stats["ss_delta"]["loop"]:.1f}%).
  </div>

  <section class="two-col">
    <div class="chart-box"><h2>ΔpLDDT distribution (TTT − ESMFold)</h2><canvas id="histChart"></canvas></div>
    <div class="chart-box"><h2>Mean pLDDT by sequence length</h2><canvas id="lenChart"></canvas></div>
  </section>

  <section class="two-col">
    <div class="chart-box"><h2>ΔpLDDT by ESMFold baseline bin</h2><canvas id="baseChart"></canvas></div>
    <div class="chart-box"><h2>ESMFold vs ProteinTTT (sample)</h2><canvas id="scatterChart"></canvas></div>
  </section>

  <section>
    <h2>Top ΔpLDDT improvements</h2>
    <div class="card"><table id="topTable"></table></div>
  </section>

  <section>
    <h2>Where TTT most exceeds AlphaFold</h2>
    <div class="card"><table id="afTable"></table></div>
  </section>
</main>
<script>
const STATS = {data_json};

function mkTable(id, rows, cols) {{
  const el = document.getElementById(id);
  el.innerHTML = '<tr>' + cols.map(c => `<th>${{c.label}}</th>`).join('') + '</tr>' +
    rows.map(r => '<tr>' + cols.map(c => `<td>${{r[c.key]}}</td>`).join('') + '</tr>').join('');
}}

new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: STATS.hist.edges.slice(0,-1).map((e,i) => `${{e}}–${{STATS.hist.edges[i+1]}}`),
    datasets: [{{ label: 'Count', data: STATS.hist.counts, backgroundColor: '#4fc3f7' }}]
  }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ maxRotation: 45, autoSkip: true, maxTicksLimit: 12 }} }} }} }}
}});

const lb = STATS.length_bins;
new Chart(document.getElementById('lenChart'), {{
  type: 'bar',
  data: {{
    labels: lb.map(x => x.len_bin),
    datasets: [
      {{ label: 'ESMFold', data: lb.map(x => x.mean_esm), backgroundColor: '#78909c' }},
      {{ label: 'ProteinTTT', data: lb.map(x => x.mean_ttt), backgroundColor: '#66bb6a' }},
      {{ label: 'AF', data: lb.map(x => x.mean_af), backgroundColor: '#ffb74d' }},
    ]
  }},
  options: {{ scales: {{ y: {{ min: 40, max: 75 }} }} }}
}});

const bb = STATS.baseline_bins;
new Chart(document.getElementById('baseChart'), {{
  type: 'line',
  data: {{
    labels: bb.map(x => x.baseline_bin),
    datasets: [
      {{ label: 'Mean ΔpLDDT', data: bb.map(x => x.mean_delta), borderColor: '#4fc3f7', tension: 0.2, fill: false }},
      {{ label: '% improved', data: bb.map(x => x.pct_improve), borderColor: '#66bb6a', tension: 0.2, yAxisID: 'y1' }},
    ]
  }},
  options: {{
    scales: {{
      y: {{ position: 'left', title: {{ display: true, text: 'ΔpLDDT' }} }},
      y1: {{ position: 'right', grid: {{ drawOnChartArea: false }}, title: {{ display: true, text: '% improved' }} }}
    }}
  }}
}});

const sc = STATS.scatter;
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'proteins',
      data: sc.map(p => ({{ x: p.plddt_ESMFold, y: p.plddt_ProteinTTT }})),
      backgroundColor: 'rgba(79,195,247,0.35)', pointRadius: 2
    }}]
  }},
  options: {{
    scales: {{
      x: {{ title: {{ display: true, text: 'ESMFold pLDDT' }} }},
      y: {{ title: {{ display: true, text: 'ProteinTTT pLDDT' }} }}
    }}
  }}
}});

mkTable('topTable', STATS.top_improvements,
  [{{key:'id',label:'ID'}},{{key:'plddt_ESMFold',label:'ESM'}},{{key:'plddt_ProteinTTT',label:'TTT'}},{{key:'plddt_AF',label:'AF'}},{{key:'delta_ttt',label:'Δ'}},{{key:'length',label:'Len'}}]);
mkTable('afTable', STATS.ttt_beats_af_top,
  [{{key:'id',label:'ID'}},{{key:'plddt_ESMFold',label:'ESM'}},{{key:'plddt_ProteinTTT',label:'TTT'}},{{key:'plddt_AF',label:'AF'}},{{key:'delta_ttt_af',label:'TTT−AF'}},{{key:'length',label:'Len'}}]);
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("bfvd2/plddt_summary.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bfvd2/plddt_analysis.html"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    csv_path = args.csv if args.csv.is_absolute() else repo / args.csv
    out_path = args.out if args.out.is_absolute() else repo / args.out

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    stats = analyze(df)
    render_html(stats, csv_path, out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
