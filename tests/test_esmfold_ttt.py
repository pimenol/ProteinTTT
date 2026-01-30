"""
Test for ProteinTTT + ESMFold.
Verifies that pLDDT improves at least 2x after applying TTT.
"""

import tempfile
from pathlib import Path

import torch
import esm
import biotite.structure.io as bsio

from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG

TEST_SEQUENCE = "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKSLIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTTLENLIATDAFAKEMVEVIIEE"


def get_plddt_from_pdb(pdb_content: str) -> float:
    """Extract mean pLDDT from PDB content using biotite."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_path = Path(f.name)
    
    try:
        struct = bsio.load_structure(str(temp_path), extra_fields=["b_factor"])
        return struct.b_factor.mean()
    finally:
        temp_path.unlink()


def test_esmfold_ttt_improves_plddt_2x():
    """
    Test that ProteinTTT improves pLDDT by at least 2x.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    
    sequence = TEST_SEQUENCE
    
    print("Loading ESMFold model...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    
    print("Running baseline inference (without TTT)...")
    with torch.no_grad():
        baseline_output = model.infer_pdb(sequence)
    baseline_plddt = get_plddt_from_pdb(baseline_output)
    print(f"Baseline pLDDT (without TTT): {baseline_plddt:.2f}")
    
    print("Applying TTT...")
    ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    
    ttt_model = ESMFoldTTT.ttt_from_pretrained(model, esmfold_config=model.cfg)
    ttt_model.ttt(sequence)
    
    print("Running inference after TTT...")
    with torch.no_grad():
        ttt_output = ttt_model.infer_pdb(sequence)
    ttt_plddt = get_plddt_from_pdb(ttt_output)
    print(f"TTT pLDDT (with TTT): {ttt_plddt:.2f}")
    
    improvement_ratio = ttt_plddt / baseline_plddt
    print(f"Improvement ratio: {improvement_ratio:.2f}x")
    
    ttt_model.ttt_reset()
    
    if improvement_ratio >= 2.0:
        print(f"✓ TEST PASSED: pLDDT improved {improvement_ratio:.2f}x (>= 2x required)")
    else:
        raise AssertionError(
            f"✗ TEST FAILED: pLDDT improvement was only {improvement_ratio:.2f}x "
            f"(baseline: {baseline_plddt:.2f}, after TTT: {ttt_plddt:.2f}). "
            f"Expected at least 2x improvement."
        )


if __name__ == "__main__":
    test_esmfold_ttt_improves_plddt_2x()
