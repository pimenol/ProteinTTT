#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def mean_plddt_from_cif(path: Path) -> float:
    """
    AlphaFold writes pLDDT into mmCIF `_atom_site.B_iso_or_equiv`.
    We compute a per-residue mean by averaging Cα B-factors.
    """
    in_atom_site_loop = False
    headers: list[str] = []
    ca_by_res: dict[str, float] = {}

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line == "loop_":
                # Start collecting headers; we'll confirm it's atom_site via first header.
                in_atom_site_loop = True
                headers = []
                continue

            if in_atom_site_loop and line.startswith("_atom_site."):
                headers.append(line)
                continue

            # If we were in some other loop, abort when data starts.
            if in_atom_site_loop and headers and not line.startswith("_atom_site."):
                # atom_site data lines begin; parse until a new CIF directive / loop starts.
                try:
                    b_idx = headers.index("_atom_site.B_iso_or_equiv")
                    atom_idx = headers.index("_atom_site.label_atom_id")
                    res_idx = headers.index("_atom_site.label_seq_id")
                except ValueError:
                    # Not the atom_site loop; keep scanning.
                    in_atom_site_loop = False
                    headers = []
                    continue

                parts = line.split()
                while True:
                    if not parts or parts[0].startswith("_") or parts[0] in {"loop_", "#"}:
                        break
                    if len(parts) >= len(headers):
                        if parts[0] == "ATOM" and parts[atom_idx] == "CA":
                            res = parts[res_idx]
                            b_raw = parts[b_idx]
                            if b_raw != "?":
                                ca_by_res[res] = float(b_raw)

                    raw = f.readline()
                    if not raw:
                        break
                    parts = raw.strip().split()

                break

    if not ca_by_res:
        raise ValueError(f"No Cα pLDDT values parsed from {path}")

    return mean(list(ca_by_res.values()))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    af_dir = repo_root / "data" / "benchmark" / "mbg" / "AF"
    summary_csv = repo_root / "data" / "benchmark" / "mbg" / "summary.csv"

    cif_paths = sorted(af_dir.glob("fold_*_model_0.cif"))
    if not cif_paths:
        raise SystemExit(f"No CIFs found under {af_dir}")

    id_from_name = re.compile(r"^fold_(?P<id>.+)_model_0\.cif$", re.IGNORECASE)
    plddt_by_id: dict[str, float] = {}
    for p in cif_paths:
        m = id_from_name.match(p.name)
        if not m:
            continue
        prot_id = m.group("id").upper()
        plddt_by_id[prot_id] = mean_plddt_from_cif(p)

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty CSV: {summary_csv}")
        fieldnames = list(reader.fieldnames)
        if "plddt_AF" not in fieldnames:
            raise SystemExit(f"`plddt_AF` column missing in {summary_csv}")
        rows = list(reader)

    updated = 0
    missing = []
    for r in rows:
        prot_id = (r.get("id") or "").strip().upper()
        if not prot_id:
            continue
        if prot_id in plddt_by_id:
            r["plddt_AF"] = f"{plddt_by_id[prot_id]:.2f}"
            updated += 1
        else:
            missing.append(prot_id)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Computed pLDDT for {len(plddt_by_id)} CIFs.")
    print(f"Updated {updated}/{len(rows)} rows in {summary_csv}.")
    if missing:
        print(f"Missing CIFs for {len(missing)} ids (showing up to 10): {missing[:10]}")


if __name__ == "__main__":
    main()

