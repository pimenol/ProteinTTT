"""Score the ESMC categorical-Jacobian contact maps of contact_jacobian.py against experimental structures.

True contacts: CB-CB (CA for Gly) < 8 A in <pdb_dir>/<id>.pdb, residues mapped to the folded
sequence by a global alignment (the chains have unobserved residues). Per protein and TTT step:
precision of the top-L (and top-L/5) scored pairs, long range (|i-j| >= 24) and medium range
(12-23), over observed pairs only (L = observed residues); plus foldseek lDDT of the step's
structure (evaluate.PdbReference). Then: does contact precision rise with lDDT?

Run on the login node in the `esmfold2` env:
    python scripts/ESMFold2/contacts_eval.py --root cameo/esmfold2_contacts --csv cameo/summary.csv --pdb_dir cameo/pdbs
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, "/scratch/project/open-35-8/pimenol1/ProteinTTT/ProteinTTT_fresh/scripts/ESMFold2")
from evaluate import THREE2ONE, PdbReference  # noqa: E402


def true_contacts(pdb: Path, seq: str, cutoff: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """(L x L bool contact matrix, L bool observed mask) in the coordinates of `seq`."""
    res = {}  # (resseq + icode) -> [name, CA, CB]
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in ("CA", "CB") and line[16] in " A":
            r = res.setdefault(line[22:27], [line[17:20], None, None])
            r[1 if line[12:16].strip() == "CA" else 2] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    res = [r for r in res.values() if r[1] is not None]
    chain = "".join(THREE2ONE.get(n, "M" if n == "MSE" else "X") for n, _, _ in res)
    xyz = np.array([cb if cb is not None else ca for _, ca, cb in res])
    aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                              open_gap_score=-5, extend_gap_score=-0.5)
    aln = aligner.align(seq, chain)[0]
    coords = np.full((len(seq), 3), np.nan)
    for (s0, s1), (c0, c1) in zip(*aln.aligned):
        coords[s0:s1] = xyz[c0:c1]
    obs = ~np.isnan(coords[:, 0])
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    return d < cutoff, obs


def precision(score: np.ndarray, contact: np.ndarray, obs: np.ndarray, lo: int, hi: int, frac: float) -> float:
    """Precision of the top (frac * L_obs) pairs with lo <= |i-j| < hi among observed pairs."""
    i, j = np.triu_indices(len(score), 1)
    keep = obs[i] & obs[j] & (j - i >= lo) & (j - i < hi)
    i, j = i[keep], j[keep]
    if not contact[i, j].any():
        return np.nan  # no true contact in this range: precision undefined
    k = max(1, min(len(i), round(frac * obs.sum())))
    top = np.argsort(-score[i, j], kind="stable")[:k]
    return float(contact[i[top], j[top]].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pdb_dir", required=True)
    args = ap.parse_args()
    root, pdb_dir = Path(args.root), Path(args.pdb_dir)
    seqs = pd.read_csv(args.csv).set_index("id").sequence
    ids = sorted(p.parent.name for p in root.glob("*/log.tsv"))

    # lDDT of every step's structure (one foldseek call per step)
    ref, lddt = PdbReference(str(pdb_dir)), {}
    with tempfile.TemporaryDirectory() as tmp:
        for step in sorted({int(f.stem.split("_")[1]) for f in root.glob("*/step_*.pdb")}):
            d = Path(tmp) / str(step)
            d.mkdir()
            for pid in ids:
                if (root / pid / f"step_{step}.pdb").exists():
                    (d / f"{pid}.pdb").symlink_to((root / pid / f"step_{step}.pdb").resolve())
            for pid, (tm, ld) in ref.compare_dir(d, ids).items():
                lddt[pid, step] = (tm, ld)

    rows = []
    for pid in ids:
        log = pd.read_csv(root / pid / "log.tsv", sep="\t").set_index("step")
        contact, obs = true_contacts(pdb_dir / f"{pid}.pdb", seqs[pid])
        for f in root.glob(f"{pid}/jac_step*.npy"):
            step = int(f.stem[len("jac_step"):])
            c = np.load(f).astype(np.float32)
            tm, ld = lddt.get((pid, step), (np.nan, np.nan))
            rows.append(dict(
                id=pid, step=step, selected=bool(log.selected[step]), length=len(seqs[pid]), observed=int(obs.sum()),
                plddt=log.plddt[step], lddt=ld, tm=tm,
                lr_PL=precision(c, contact, obs, 24, 10**6, 1.0), lr_PL5=precision(c, contact, obs, 24, 10**6, 0.2),
                mr_PL=precision(c, contact, obs, 12, 24, 1.0), sr_PL=precision(c, contact, obs, 6, 12, 1.0),
                all_PL=precision(c, contact, obs, 6, 10**6, 1.0),
            ))
    d = pd.DataFrame(rows).sort_values(["id", "step"])
    d.to_csv(root / "per_protein_step.csv", index=False)

    metrics = ["lr_PL", "lr_PL5", "mr_PL", "all_PL"]
    base = d[d.step == 0].set_index("id")
    sel = d[d.selected].set_index("id")
    print(f"{len(ids)} proteins; step 0 vs the step selected by pLDDT")
    for m in ["lddt", "plddt"] + metrics:
        b, s = base[m], sel[m].reindex(base.index)
        ok = b.notna() & s.notna()
        diff = (s - b)[ok]
        p = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        print(f"  {m:7s} n={ok.sum():2d}  {b[ok].mean():.3f} -> {s[ok].mean():.3f}  ({diff.mean():+.3f}; "
              f"{(diff > 0).sum()} up / {(diff < 0).sum()} down; Wilcoxon p={p:.3g})")
    dl = (sel.lddt.reindex(base.index) - base.lddt)
    print("Spearman across proteins, delta(metric) vs delta(lDDT), selected - step 0:")
    for m in metrics:
        dm = sel[m].reindex(base.index) - base[m]
        ok = dm.notna() & dl.notna()
        r = spearmanr(dm[ok], dl[ok])
        print(f"  {m:7s} n={ok.sum():2d}  rho={r.statistic:+.2f}  p={r.pvalue:.3g}")
    print("Mean by step (fixed steps):")
    print(d[~d.selected | d.step.isin([0, 5, 10, 15, 20])].groupby("step")[["lddt", "plddt"] + metrics]
          .mean().round(3).to_string())
    print("Within-protein Spearman over the steps, metric vs lDDT (mean over proteins):")
    for m in metrics:
        rs = [spearmanr(g[m], g.lddt).statistic for _, g in d.groupby("id") if g[m].nunique() > 1 and g.lddt.nunique() > 1]
        print(f"  {m:7s} n={len(rs):2d}  mean rho={np.mean(rs):+.2f}  ({sum(r > 0 for r in rs)} positive)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    dm = sel.lr_PL.reindex(base.index) - base.lr_PL
    ax[0].scatter(dl, dm)
    ax[0].axhline(0, c="gray", lw=0.5)
    ax[0].axvline(0, c="gray", lw=0.5)
    ax[0].set(xlabel="ΔlDDT (selected − step 0)", ylabel="Δ long-range P@L (ESMC Jacobian)")
    by = d[d.step.isin([0, 5, 10, 15, 20])].groupby("step")[["lddt", "lr_PL"]].mean()
    ax[1].plot(by.index, by.lddt, "o-", label="lDDT")
    ax[1].plot(by.index, by.lr_PL, "s-", label="long-range P@L")
    ax[1].set(xlabel="TTT step")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(root / "contacts_vs_lddt.png", dpi=150)


if __name__ == "__main__":
    main()
