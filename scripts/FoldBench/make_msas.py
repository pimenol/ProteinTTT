"""Fetch one MMseqs2 (ColabFold server) .a3m per unique FoldBench protein sequence into foldbench/msa/."""
import glob
import json
import sys
from pathlib import Path

from proteinttt.utils.msa import MSAServer

root = Path(sys.argv[1] if len(sys.argv) > 1 else "foldbench")
seqs = {}
for f in sorted(glob.glob(str(root / "inputs/*.json"))):
    for c in json.load(open(f))["chains"]:
        if c["type"] == "protein":
            seqs[Path(c["msa"]).stem] = c["sequence"]
server = MSAServer(cache_dir=root / "msa")
for i, (sid, seq) in enumerate(sorted(seqs.items(), key=lambda kv: len(kv[1]))):
    try:
        server.get(seq=seq, seq_id=sid)
        print(f"{i + 1}/{len(seqs)} {sid} ok", flush=True)
    except Exception as e:
        print(f"{i + 1}/{len(seqs)} {sid} FAILED {e!r}", flush=True)
