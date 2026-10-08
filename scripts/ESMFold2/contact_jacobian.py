"""Unsupervised contact maps from ESMC (categorical Jacobian) along a ProteinTTT run.

For every protein: run ESMFold2 + ProteinTTT with the given config (same seeding as
run_dataset.py) and compute the categorical Jacobian of the ESMC masked-LM logits
(Zhang et al. 2024) at --steps and at the step ProteinTTT selects by pLDDT.
Forward passes only: position i is mutated to each of the 20 amino acids, the change of the
logits at every position j is J[i, a, j, b]; contact score = APC-corrected Frobenius norm.

Output <out>/<id>/: jac_step<N>.npy (L x L, float16), step_<N>.pdb (all TTT steps),
lora_sel.pt (LoRA weights of the selected step), log.tsv (per-step pLDDT / times, `selected`).
Scoring against the experimental structure is done offline by contacts_eval.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from proteinttt.models.esmfold2 import DEFAULT_ESMFOLD2_TTT_CFG, ESMFold2TTT, load_esmfold2

AA = list(range(4, 24))  # the 20 standard amino acids in the ESMC vocabulary
# keys of the run_dataset.py YAML that are not TTTConfig fields
SCRIPT_ONLY_KEYS = {"df_path", "output", "input", "compute_step_metrics", "new_experement_dir", "columns",
                    "generate_msa", "describe_structure", "compute_ss_percentages", "save_results_table", "seed",
                    "rerun_helix", "rerun_seed_offset", "helix_pct_min", "helix_dominance_min", "helix_plddt_min",
                    "model", "esmfold2_models"}


@torch.no_grad()
def categorical_jacobian(model: ESMFold2TTT, x: torch.Tensor, tokens_per_batch: int) -> torch.Tensor:
    """L x L contact scores for the tokenised sequence x ([1, L + 2], with <cls>/<eos>)."""
    L = x.shape[1] - 2
    aa = torch.tensor(AA, device=x.device)
    wt = model._ttt_predict_logits(x)[0, 1:-1][:, aa]  # [L, 20]
    jac = torch.empty(L * 20, L, 20, device=x.device)
    pos = torch.arange(L, device=x.device).repeat_interleave(20)  # mutated position of every variant
    new = aa.repeat(L)
    bs = max(20, tokens_per_batch // (L + 2))
    for s in range(0, L * 20, bs):
        batch = x.repeat(len(pos[s:s + bs]), 1)
        batch[torch.arange(len(batch)), 1 + pos[s:s + bs]] = new[s:s + bs]
        jac[s:s + bs] = model._ttt_predict_logits(batch)[:, 1:-1][:, :, aa] - wt
    jac = jac.view(L, 20, L, 20)
    for dim in range(4):
        jac = jac - jac.mean(dim, keepdim=True)
    jac = (jac + jac.permute(2, 3, 0, 1)) / 2
    c = jac.pow(2).sum((1, 3)).sqrt()
    c.fill_diagonal_(0)
    c = c - c.sum(0, keepdim=True) * c.sum(1, keepdim=True) / c.sum()  # APC
    c.fill_diagonal_(0)
    return c


class JacobianTTT(ESMFold2TTT):
    """ESMFold2TTT that also saves the categorical Jacobian at selected TTT steps."""

    jac_steps: tuple = ()
    jac_dir: Path = None
    jac_tokens: int = 16000

    def save_jacobian(self, step: int, seq: str) -> None:
        x = self._ttt_tokenize(seq).to(next(self.parameters()).device)
        torch.cuda.synchronize()
        t0 = time.time()
        c = categorical_jacobian(self, x, self.jac_tokens)
        torch.cuda.synchronize()
        self.jac_time[step] = time.time() - t0
        np.save(self.jac_dir / f"jac_step{step}.npy", c.cpu().numpy().astype(np.float16))

    def _ttt_eval_step(self, step: int, seq: str, **kwargs):
        out = super()._ttt_eval_step(step=step, seq=seq, **kwargs)
        if step in self.jac_steps:
            self.save_jacobian(step, seq)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--msa_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="0,5,10,15,20")
    ap.add_argument("--ids", default=None, help="comma-separated ids (default: all rows)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--tokens", type=int, default=16000, help="tokens per Jacobian forward batch")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config.get("seed", 0))
    ttt_cfg = DEFAULT_ESMFOLD2_TTT_CFG
    for key, value in config.items():
        if key not in SCRIPT_ONLY_KEYS:
            setattr(ttt_cfg, key, value)
    df = pd.read_csv(args.csv).iloc[args.start:args.end]
    if args.ids:
        df = df[df.id.isin(args.ids.split(","))]

    model = JacobianTTT(ttt_cfg, *(m.cuda() for m in load_esmfold2(**config.get("esmfold2_models", {}))))
    model.jac_steps = tuple(int(s) for s in args.steps.split(","))
    model.jac_tokens = args.tokens

    for _, p in df.iterrows():
        out = Path(args.out) / p.id
        if (out / "log.tsv").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        model.jac_dir, model.jac_time = out, {}
        try:
            # same seeding as run_dataset.py (ttt_pass)
            model.ttt_generator.manual_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model.ttt_cfg.seed = seed
            model.ttt_reset()
            t0 = time.time()
            res = model.ttt(p.sequence, msa_pth=Path(args.msa_dir) / f"{p.id}.a3m")
            log = res["df"].copy()
            sel = int(log.loc[log.plddt.idxmax(), "step"])  # the state ttt() leaves the model in
            if sel not in model.jac_steps:
                model.save_jacobian(sel, p.sequence)
            torch.save(model._ttt_get_state(), out / "lora_sel.pt")
            for step, data in res["ttt_step_data"].items():
                (out / f"step_{step}.pdb").write_text(data["eval_step_preds"]["pdb"])
        except Exception as e:  # keep going, log the protein as failed
            print(f"Error for {p.id}: {e!r}", flush=True)
            torch.cuda.empty_cache()
            continue
        log["selected"] = log.step == sel
        log["jac_time"] = log.step.map(model.jac_time)
        log.to_csv(out / "log.tsv", sep="\t", index=False)
        print(f"{p.id} L={len(p.sequence)}: pLDDT {log.plddt[0]:.1f} -> {log.plddt.max():.1f} (step {sel}), "
              f"total {time.time() - t0:.0f}s, Jacobian {sum(model.jac_time.values()):.0f}s "
              f"for {len(model.jac_time)} states", flush=True)


if __name__ == "__main__":
    main()
