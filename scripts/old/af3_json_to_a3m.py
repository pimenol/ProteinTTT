"""Convert AlphaFold3 server JSON files to .a3m MSA files for ProteinTTT.

Usage:
    python af3_json_to_a3m.py \
        --json_dir /path/to/af3_data_jsons \
        --output_dir /path/to/data/marts_db/msa \
        --summary /path/to/data/marts_db/summary.csv
"""

import argparse
import json
import csv
import sys
from pathlib import Path


def extract_msa_from_af3_json(json_path: Path) -> tuple[str, str, str]:
    """Return (name, query_sequence, msa_a3m_string) from an AF3 JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    name = data.get("name", json_path.stem)

    for entry in data.get("sequences", []):
        prot = entry.get("protein") or entry.get("proteinChain")
        if prot is None:
            continue
        seq = prot.get("sequence", "")
        msa = prot.get("unpairedMsa", "")
        if msa:
            return name, seq, msa

    return name, "", ""


def build_sequence_to_id_map(summary_path: Path) -> dict[str, str]:
    """Map sequence → id from the summary CSV."""
    seq_to_id = {}
    with open(summary_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["sequence"].strip().upper()
            seq_to_id[seq] = row["id"]
    return seq_to_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert AF3 server JSON files to .a3m MSA files."
    )
    parser.add_argument(
        "--json_dir", type=str, required=True,
        help="Directory containing AF3 JSON files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for .a3m files."
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Path to summary.csv to map sequences to IDs."
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_to_id = {}
    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.exists():
            print(f"Error: summary file not found: {summary_path}")
            sys.exit(1)
        seq_to_id = build_sequence_to_id_map(summary_path)
        print(f"Loaded {len(seq_to_id)} sequences from {summary_path}")

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")

    converted = 0
    skipped = 0
    for jf in json_files:
        name, seq, msa_str = extract_msa_from_af3_json(jf)
        if not msa_str:
            print(f"  SKIP (no MSA): {jf.name}")
            skipped += 1
            continue

        # Determine output ID: match via sequence, fall back to JSON "name" field
        if seq_to_id:
            seq_clean = seq.strip().upper()
            file_id = seq_to_id.get(seq_clean)
            if file_id is None:
                # Try extracting ID from the JSON name (e.g. "marts_E00010" -> "E00010")
                parts = name.split("_", 1)
                file_id = parts[1] if len(parts) > 1 else name
                print(f"  WARN: sequence from {jf.name} not in summary, using name '{file_id}'")
        else:
            parts = name.split("_", 1)
            file_id = parts[1] if len(parts) > 1 else name

        out_path = output_dir / f"{file_id}.a3m"
        with open(out_path, "w") as f:
            f.write(msa_str)
        converted += 1

    print(f"\nDone: {converted} converted, {skipped} skipped out of {len(json_files)} files.")


if __name__ == "__main__":
    main()
