"""Apply logit Platt scaling to ESMFold+ProteinTTT pLDDT and write a summary CSV.

    pLDDT_cal = 100 * sigmoid(a * logit(pLDDT / 100) + b)

Writes PDBs with calibrated B-factors to <ttt_dir>_calibrated and a per-protein
summary CSV with raw and calibrated mean pLDDT.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

A, B = 0.5376, -0.1063
EPS = 1e-4


def calibrate(plddt):
    x = np.clip(np.asarray(plddt, dtype=float) / 100, EPS, 1 - EPS)
    return 100 / (1 + np.exp(-(A * np.log(x / (1 - x)) + B)))


def is_atom(line):
    return line.startswith(("ATOM", "HETATM"))


def mean_plddt(pdb_path):
    with open(pdb_path) as f:
        b = [float(line[60:66]) for line in f if is_atom(line)]
    return float(np.mean(b)) if b else np.nan


def calibrate_pdb(src, dst):
    """Rewrite B-factors with calibrated pLDDT; return (raw mean, calibrated mean)."""
    lines = src.read_text().splitlines(keepends=True)
    idx = [i for i, line in enumerate(lines) if is_atom(line)]
    raw = np.array([float(lines[i][60:66]) for i in idx])
    cal = calibrate(raw)
    for i, v in zip(idx, cal):
        lines[i] = f"{lines[i][:60]}{v:6.2f}{lines[i][66:]}"
    dst.write_text("".join(lines))
    return float(raw.mean()), float(cal.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--summary_file", type=Path, required=True)
    p.add_argument("--output_csv", type=Path, default=None)
    args = p.parse_args()

    pred = args.run_dir / "predicted_structures"
    ttt_dir = pred / "ESMFold_ProteinTTT"
    out_dir = pred / "ESMFold_ProteinTTT_calibrated"
    out_dir.mkdir(exist_ok=True)
    output_csv = args.output_csv or args.run_dir / "plddt_summary.csv"

    rows = []
    for src in sorted(ttt_dir.glob("*.pdb")):
        raw, cal = calibrate_pdb(src, out_dir / src.name)
        base = pred / "ESMFold" / src.name
        rows.append({
            "id": src.stem,
            "plddt_ESMFold": mean_plddt(base) if base.exists() else np.nan,
            "plddt_ProteinTTT": raw,
            "plddt_ProteinTTT_calibrated": cal,
        })

    meta = pd.read_csv(args.summary_file).rename(columns={"avg_plddt": "plddt_AF"})
    df = pd.DataFrame(rows).merge(meta, on="id", how="left")
    df.to_csv(output_csv, index=False)

    print(f"{len(df)} proteins -> {output_csv}")
    print(f"calibrated PDBs -> {out_dir}")
    print(f"missing metadata: {df['sequence'].isna().sum()}")
    cols = ["plddt_AF", "plddt_ESMFold", "plddt_ProteinTTT",
            "plddt_ProteinTTT_calibrated"]
    print(df[cols].describe().round(2).to_string())
    for c in cols:
        print(f"{c}: {(df[c] >= 70).mean() * 100:.1f}% >= 70")


if __name__ == "__main__":
    main()
