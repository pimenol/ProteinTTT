"""Write FoldBench prediction_reference.csv for one run dir and file name, then call FoldBench evaluate.py.

Run in the `foldbench` conda env from the repo root:
  python scripts/FoldBench/score_foldbench.py --run foldbench/esmfold2_msa --file baseline.cif --name ESMFold2_MSA
  python scripts/FoldBench/score_foldbench.py --run foldbench/ttt --file ttt.cif --name ESMFold2_MSA_TTT
Outputs land in foldbench/evaluation/<name>/ (raw/*.csv has per-interface DockQ, LRMSD, LDDT-PLI).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

TASKS = ["interface_protein_dna", "interface_protein_rna", "interface_protein_ligand"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--file", required=True, help="baseline.cif or ttt.cif")
    p.add_argument("--name", required=True)
    p.add_argument("--root", default="foldbench")
    args = p.parse_args()
    root, run = Path(args.root).resolve(), Path(args.run).resolve()

    rows = []
    for d in sorted(run.iterdir()):
        cif, rj = d / args.file, d / "row.json"
        if not (cif.exists() and rj.exists()):
            continue
        r = json.loads(rj.read_text())
        score = r.get("plddt_best" if args.file == "ttt.cif" else "plddt", r.get("plddt_step0"))
        rows.append({"pdb_id": d.name, "seed": 0, "sample": 0, "ranking_score": score, "prediction_path": str(cif)})
    ev = root / "evaluation" / args.name
    ev.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ev / "prediction_reference.csv", index=False)
    print(f"{len(rows)} predictions -> {ev / 'prediction_reference.csv'}")

    # FoldBench merges targets with predictions (left join); keep only predicted entries
    tdir = ev / "targets"
    tdir.mkdir(exist_ok=True)
    ids = {r["pdb_id"] for r in rows}
    tasks = []
    for t in TASKS:
        df = pd.read_csv(root / "targets" / f"{t}.csv")
        df = df[df.pdb_id.isin(ids)]
        df.to_csv(tdir / f"{t}.csv", index=False)
        if len(df):
            tasks.append(t)

    fb = root / "FoldBench"
    common = ["--evaluation_dir", str(root / "evaluation"), "--algorithm_name", args.name]
    subprocess.run(
        [sys.executable, "evaluate.py", "--targets_dir", str(tdir), "--ground_truth_dir", str(root / "ground_truths"),
         *common, "--targets", *tasks], cwd=fb, check=True,
    )
    subprocess.run(
        [sys.executable, "task_score_summary.py", "--evaluation_dir", str(root / "evaluation"), "--target_dir", str(tdir),
         "--output_path", str(ev / "summary_table.csv"), "--algorithm_names", args.name, "--targets", *tasks,
         "--metric_type", "rank"], cwd=fb, check=True,
    )
    print(pd.read_csv(ev / "summary_table.csv").to_string())


if __name__ == "__main__":
    main()
