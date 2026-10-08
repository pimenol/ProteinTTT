"""ESMFold2 + ProteinTTT: fold with averaged / interpolated LoRA weights instead of the pLDDT-selected step.

One normal TTT run per protein (config = a run_dataset.py YAML); the LoRA weights of every step are kept
in RAM, and after the last step the protein is folded again with:
  avg_A_B          exact mean of the LoRA updates dW = up @ down over steps A..B (rank concatenation)
  avg_top5_plddt   same, over the 5 steps with the highest pLDDT
  avgAB_11_20      naive mean of `up` and `down` separately, steps 11..20
  interp_X_tNN     W0 + t * dW, t = 0.25 / 0.5 / 0.75, X = last step / pLDDT-selected step / avg_11_20

Every step and variant is saved as <out>/<id>/<name>.pdb and logged in samples.tsv (pLDDT, pTM, time),
the format of sample_baseline.py, so eval_sampling.score_protein scores them.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from proteinttt.models.esmfold2 import DEFAULT_ESMFOLD2_TTT_CFG, ESMFold2TTT, load_esmfold2

State = dict[str, torch.Tensor]


def mean_delta(states: list[State]) -> State:
    """Exact mean of up @ down: stack the ranks, [up_1 .. up_k] / k @ [down_1; ..; down_k]."""
    return {
        n: torch.cat([s[n] for s in states], dim=0) if "lora_down" in n
        else torch.cat([s[n] for s in states], dim=1) / len(states)
        for n in states[0]
    }


def mean_ab(states: list[State]) -> State:
    return {n: torch.stack([s[n] for s in states]).mean(0) for n in states[0]}


def scaled(state: State, t: float) -> State:
    return {n: v * t if "lora_up" in n else v for n, v in state.items()}


def variants(states: list[State], plddt: np.ndarray, check: bool) -> dict[str, State]:
    last = len(states) - 1
    avg10 = mean_delta(states[last - 9:])
    out = {
        f"avg_{last - 2}_{last}": mean_delta(states[last - 2:]),
        f"avg_{last - 4}_{last}": mean_delta(states[last - 4:]),
        f"avg_{last - 9}_{last}": avg10,
        f"avg_1_{last}": mean_delta(states[1:]),
        "avg_top5_plddt": mean_delta([states[i] for i in np.argsort(-plddt)[:5]]),
        f"avgAB_{last - 9}_{last}": mean_ab(states[last - 9:]),
    }
    bases = {"last": states[last], "sel": states[int(np.argmax(plddt))], "avg10": avg10}
    for name, base in bases.items():
        for t in (25, 50, 75):
            out[f"interp_{name}_t{t}"] = scaled(base, t / 100)
    if check:  # must reproduce the last step
        out["check_last"] = mean_delta([states[last]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated ids (smoke test)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    df = pd.read_csv(Path(config["df_path"]) / config["input"]["summary_file"]).iloc[args.start:args.end]
    df = df[df.sequence.str.len() <= config.get("max_sequence_length", 500)]
    if args.ids:
        df = df[df.id.isin(args.ids.split(","))]
    msa_dir = Path(config["input"]["msa_dir"])

    ttt_cfg = DEFAULT_ESMFOLD2_TTT_CFG
    for key, value in config.items():
        if key not in {"seed", "model", "esmfold2_models"}:
            setattr(ttt_cfg, key, value)
    ttt_cfg.keep_step_states = True
    model = ESMFold2TTT(ttt_cfg, *(m.cuda() for m in load_esmfold2(**config.get("esmfold2_models", {}))))

    for _, p in df.iterrows():
        out = Path(args.out) / p.id
        if (out / "samples.tsv").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        seq = p.sequence.strip().upper()
        print(f"{p.id} (length {len(seq)})", flush=True)

        # Same per-protein seeding as run_dataset.py
        model.ttt_generator.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model.ttt_cfg.seed = args.seed
        model.ttt_reset()
        res = model.ttt(seq, msa_pth=msa_dir / f"{p.id}.a3m")
        log = res["df"]
        log.to_csv(out / "ttt_log.tsv", sep="\t", index=False)

        rows = []
        for step, entry in res["ttt_step_data"].items():
            (out / f"step_{step}.pdb").write_text(entry["eval_step_preds"]["pdb"])
            r = log.iloc[step]
            rows.append(dict(sample=f"step_{step}", plddt=r.plddt, ptm=r.ptm, time=r.eval_step_time))

        params = dict(model.named_parameters())
        for name, state in variants(model.step_states, log.plddt.to_numpy(), args.check).items():
            for n, v in state.items():
                params[n].data = v.cuda()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                r = model.infer(seq)
            torch.cuda.synchronize()
            (out / f"{name}.pdb").write_text(r.complex.to_protein_complex().to_pdb_string())
            rows.append(dict(sample=name, plddt=100 * float(r.plddt.mean()), ptm=float(r.ptm), time=time.time() - t0))
            print(f"  {name}: pLDDT {rows[-1]['plddt']:.2f}", flush=True)
        model.step_states = []
        pd.DataFrame(rows).to_csv(out / "samples.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
