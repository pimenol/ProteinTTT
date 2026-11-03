"""Build ESMFold2 complex inputs for the FoldBench protein-DNA/RNA/ligand interface targets.

Reads foldbench/targets/interface_protein_{dna,rna,ligand}.csv and the RCSB assembly CIFs in
foldbench/ground_truths/. Chains follow the FoldBench AF3 inputs: every polymer chain (by
label_asym_id, full entity sequence, canonical letters, so modified residues become their parent)
and every non-water non-polymer chain as CCD ligand(s). Writes foldbench/inputs/<pdb_id>.json and
foldbench/entries.csv (one row per entry; the TTT chain is the protein chain of the first interface row).

Usage: python scripts/FoldBench/prepare_foldbench.py --root foldbench
"""
import argparse
import hashlib
import json
from pathlib import Path

import biotite.structure.io.pdbx as px
import pandas as pd

POLY = {"polypeptide(L)": "protein", "polydeoxyribonucleotide": "dna", "polyribonucleotide": "rna"}
# Heavy-atom counts are only for the token estimate; ESMFold2 uses one token per ligand atom.
HEAVY = {"H", "D"}


def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()[:16]


def parse_entry(cif: Path) -> list[dict]:
    b = px.CIFFile.read(str(cif)).block
    ent_type = dict(zip(b["entity"]["id"].as_array(), b["entity"]["type"].as_array()))
    ep = b["entity_poly"]
    poly = {
        e: (t, s.replace("\n", ""))
        for e, t, s in zip(
            ep["entity_id"].as_array(), ep["type"].as_array(), ep["pdbx_seq_one_letter_code_can"].as_array()
        )
    }
    a = b["atom_site"]
    cols = zip(
        a["label_asym_id"].as_array(), a["label_entity_id"].as_array(), a["label_comp_id"].as_array(),
        a["label_seq_id"].as_array(), a["auth_seq_id"].as_array(), a["type_symbol"].as_array(),
        a["pdbx_PDB_model_num"].as_array(),
    )
    chains, order = {}, []
    first_model = None
    for asym, ent, comp, lseq, aseq, el, model in cols:
        first_model = first_model or model
        if model != first_model:
            continue
        if asym not in chains:
            chains[asym] = {"entity": ent, "residues": [], "heavy": 0}
            order.append(asym)
        c = chains[asym]
        key = (lseq, aseq, comp)
        if not c["residues"] or c["residues"][-1] != key:
            c["residues"].append(key)
        if el.upper() not in HEAVY:
            c["heavy"] += 1
    out = []
    for asym in order:
        c = chains[asym]
        et = ent_type.get(c["entity"])
        if et == "water":
            continue
        if et == "polymer":
            ptype, seq = poly[c["entity"]]
            if ptype not in POLY:
                raise ValueError(f"unsupported polymer type {ptype} in {asym}")
            out.append({"id": asym, "type": POLY[ptype], "sequence": seq})
        else:  # non-polymer or branched (glycans): CCD codes in residue order
            out.append({"id": asym, "type": "ligand", "ccd": [r[2] for r in c["residues"]], "heavy": c["heavy"]})
    return out


def n_tokens(chains: list[dict]) -> int:
    return sum(len(c["sequence"]) if c["type"] != "ligand" else c["heavy"] for c in chains)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="foldbench")
    args = p.parse_args()
    root = Path(args.root)
    (root / "inputs").mkdir(exist_ok=True)

    rows = []
    for task in ["rna", "dna", "ligand"]:
        df = pd.read_csv(root / "targets" / f"interface_protein_{task}.csv")
        c1, c2 = df.columns[1], df.columns[2]
        t1, t2 = df.columns[3], df.columns[4]
        for r in df.itertuples(index=False):
            r = r._asdict()
            prot = r[c1] if r[t1] == "protein" else r[c2]
            rows.append({"pdb_id": r["pdb_id"], "task": task, "protein_chain": prot})
    targets = pd.DataFrame(rows)

    entries, failed = [], []
    for pdb_id, g in targets.groupby("pdb_id", sort=False):
        try:
            chains = parse_entry(root / "ground_truths" / f"{pdb_id}.cif")
        except Exception as e:  # report and skip, never guess
            failed.append((pdb_id, repr(e)))
            continue
        by_id = {c["id"]: c for c in chains}
        ttt_chain = g["protein_chain"].iloc[0]
        if ttt_chain not in by_id or by_id[ttt_chain]["type"] != "protein":
            failed.append((pdb_id, f"TTT chain {ttt_chain} not a protein chain"))
            continue
        for c in chains:
            if c["type"] == "protein":
                c["msa"] = f"msa/{seq_hash(c['sequence'])}.a3m"
        (root / "inputs" / f"{pdb_id}.json").write_text(json.dumps({"name": pdb_id, "ttt_chain": ttt_chain, "chains": chains}))
        types = [c["type"] for c in chains]
        entries.append({
            "pdb_id": pdb_id,
            "tasks": "+".join(sorted(set(g["task"]))),
            "n_interfaces": len(g),
            "ttt_chain": ttt_chain,
            "ttt_len": len(by_id[ttt_chain]["sequence"]),
            "n_protein": types.count("protein"), "n_dna": types.count("dna"), "n_rna": types.count("rna"),
            "n_ligand": types.count("ligand"),
            "n_tokens": n_tokens(chains),
        })
    ent = pd.DataFrame(entries).sort_values("n_tokens").reset_index(drop=True)
    ent.to_csv(root / "entries.csv", index=False)
    pd.DataFrame(failed, columns=["pdb_id", "error"]).to_csv(root / "prepare_failed.csv", index=False)
    print(f"{len(ent)} entries, {len(failed)} failed")
    print(ent["n_tokens"].describe())
    for lim in [500, 1000, 1500, 2000, 3000]:
        print(f"<= {lim} tokens: {(ent.n_tokens <= lim).sum()}")


if __name__ == "__main__":
    main()
