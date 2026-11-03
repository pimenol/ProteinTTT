"""ESMFold2 (+ MSA input) and ESMFold2 + ProteinTTT_MSA on FoldBench complexes.

--screen: one fold per entry with the base model (= TTT step 0), writes <out>/<pdb_id>/baseline.cif + row.
default : full TTT on the entry's interface protein chain; all chains are folded at every step.

Usage (GPU): python scripts/FoldBench/run_foldbench.py --config scripts/ESMFold2/M_alpha256_msainput.yaml \
    --root foldbench --entries foldbench/entries.csv --start 0 --end 50 --out foldbench/esmfold2_msa [--screen]
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from proteinttt.models.esmfold2 import DEFAULT_ESMFOLD2_TTT_CFG, ESMFold2TTT, load_esmfold2

# Same keys that scripts/run_dataset.py keeps out of the TTT config
SCRIPT_ONLY_KEYS = {
    "df_path", "output", "input", "compute_step_metrics", "new_experement_dir", "columns", "generate_msa",
    "describe_structure", "compute_ss_percentages", "save_results_table", "seed", "rerun_helix",
    "rerun_seed_offset", "helix_pct_min", "helix_dominance_min", "helix_plddt_min", "model", "esmfold2_models",
    "max_sequence_length", "use_true_pdb", "use_chain_name", "renumber_pdb",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--root", default="foldbench")
    p.add_argument("--entries", required=True, help="CSV with a pdb_id column")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--screen", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    logging.getLogger("ttt_log").propagate = False

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(open(args.config))
    ttt_cfg = DEFAULT_ESMFOLD2_TTT_CFG
    for k, v in config.items():
        if k not in SCRIPT_ONLY_KEYS:
            setattr(ttt_cfg, k, v)
    ttt_cfg.seed = args.seed
    logging.info(f"TTT config: {ttt_cfg}")

    device = torch.device("cuda")
    model = ESMFold2TTT(ttt_cfg, *(m.to(device) for m in load_esmfold2(**config.get("esmfold2_models", {}))))

    ids = pd.read_csv(args.entries)["pdb_id"].tolist()[args.start:args.end]
    logging.info(f"{len(ids)} entries, rows [{args.start}, {args.end})")
    for pdb_id in ids:
        d = out / pdb_id
        if (d / "row.json").exists():
            continue
        d.mkdir(exist_ok=True)
        entry = json.load(open(root / "inputs" / f"{pdb_id}.json"))
        tc = next(c for c in entry["chains"] if c["id"] == entry["ttt_chain"])
        seq, msa_pth = tc["sequence"], root / tc["msa"]
        model.set_complex(entry, msa_root=root)
        model.ttt_generator.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model.ttt_reset()
        t0 = time.time()
        try:
            if args.screen:
                with torch.no_grad():
                    res = model.infer(seq)
                (d / "baseline.cif").write_text(res.complex.to_mmcif())
                row = {
                    "plddt": 100 * float(res.plddt.mean()), "ptm": float(res.ptm),
                    "iptm": float(res.iptm) if res.iptm is not None else None,
                }
            else:
                r = model.ttt(seq, msa_pth=msa_pth)
                df = r["df"].copy()
                df.to_csv(d / "log.tsv", sep="\t", index=False)
                for step, s in r["ttt_step_data"].items():
                    cif = s.get("eval_step_preds", {}).get("cif")
                    if cif is not None:
                        (d / f"step_{step}.cif").write_text(cif)
                (d / "baseline.cif").write_text(r["ttt_step_data"][0]["eval_step_preds"]["cif"])
                best = int(df.loc[df["plddt"].idxmax(), "step"])
                (d / "ttt.cif").write_text((d / f"step_{best}.cif").read_text())
                row = {
                    "plddt_step0": float(df["plddt"].iloc[0]), "plddt_best": float(df["plddt"].max()),
                    "best_step": best,
                    "iptm_step0": float(df["iptm"].iloc[0]), "iptm_best": float(df.loc[df["plddt"].idxmax(), "iptm"]),
                }
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"{pdb_id}: OOM {e}")
            torch.cuda.empty_cache()
            row = {"error": "OOM"}
        except Exception as e:  # keep going, record the failure
            logging.exception(f"{pdb_id}: failed")
            row = {"error": repr(e)[:300]}
        row.update({"pdb_id": pdb_id, "seconds": time.time() - t0})
        (d / "row.json").write_text(json.dumps(row))
        logging.info(f"{pdb_id}: {row}")


if __name__ == "__main__":
    main()
