"""Score weight_avg.py outputs by lDDT vs the experimental chain and compare every averaging /
interpolation rule with ProteinTTT's pLDDT step selection on the same TTT trajectory.

Usage (proteinttt env, LD_LIBRARY_PATH=$CONDA_PREFIX/lib; scripts/ESMFold2 of the main checkout on PYTHONPATH):
    python eval_weight_avg.py --root cameo/esmfold2_wavg/seed0 --pdb_dir cameo/pdbs
Writes <root>/<id>/samples_scored.tsv, <root>/per_protein.csv and <root>/summary.csv.
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from eval_sampling import score_protein
from evaluate import PdbReference


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pdb_dir", default=None, help="reference chains <dir>/<id>.pdb (e.g. CAMEO)")
    args = ap.parse_args()
    root, ref = Path(args.root), PdbReference(args.pdb_dir)

    per_protein = {}
    for d in sorted(p.parent for p in root.glob("*/samples.tsv")):
        s = score_protein(d, ref)
        if s is None or s.lddt.isna().all():
            continue
        s = s.set_index("sample")
        steps = s[s.index.str.startswith("step_")]
        row = s.lddt[~s.index.str.startswith("step_")].to_dict()
        row["step_0"] = steps.lddt.iloc[0]
        row["step_last"] = steps.lddt.iloc[-1]
        row["step_sel_plddt"] = steps.lddt[steps.plddt.idxmax()]
        row["step_oracle"] = steps.lddt.max()
        row["pool_sel_plddt"] = s.lddt[s.plddt.idxmax()]  # pLDDT pick among steps + variants
        row["pool_oracle"] = s.lddt.max()
        per_protein[d.name] = row
    df = pd.DataFrame(per_protein).T
    df.to_csv(root / "per_protein.csv", index_label="id")

    base = df.step_sel_plddt
    rows = []
    for name in df.columns:
        d = df[name] - base
        rows.append(dict(
            rule=name, lddt=df[name].mean(), delta_vs_sel=d.mean(),
            better=int((d > 0.01).sum()), worse=int((d < -0.01).sum()), n=len(df),
            p=wilcoxon(df[name], base).pvalue if (d != 0).any() else float("nan"),
        ))
    out = pd.DataFrame(rows).sort_values("lddt", ascending=False)
    out.to_csv(root / "summary.csv", index=False)
    print(out.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
