"""
Test for ProteinTTT + ESMFold.
Verifies that pLDDT improves at least 2x after applying TTT.
"""

import tempfile
from pathlib import Path

import torch
import esm
import biotite.structure.io as bsio
import proteinttt
print(proteinttt.__file__)
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
    
    ttt_model = ESMFoldTTT.ttt_from_pretrained(model, ttt_cfg=ttt_cfg, esmfold_config=model.cfg)
    ttt_model.ttt(sequence)
    
    print("Running inference after TTT...")
    with torch.no_grad():
        ttt_output = ttt_model.infer_pdb(sequence)
    ttt_plddt = get_plddt_from_pdb(ttt_output)
    print(f"TTT pLDDT (with TTT): {ttt_plddt:.2f}")
    
    improvement_ratio = ttt_plddt / baseline_plddt
    print(f"Improvement ratio: {improvement_ratio:.2f}x")
    
    model.ttt_reset()
    
    if improvement_ratio >= 2.0:
        print(f"✓ TEST PASSED: pLDDT improved {improvement_ratio:.2f}x (>= 2x required)")
    else:
        raise AssertionError(
            f"✗ TEST FAILED: pLDDT improvement was only {improvement_ratio:.2f}x "
            f"(baseline: {baseline_plddt:.2f}, after TTT: {ttt_plddt:.2f}). "
            f"Expected at least 2x improvement."
        )


def test_esmfold_ttt_reset():
    """
    Test that ttt_reset() properly restores the model to its initial state.
    
    This test verifies that:
    1. Model predictions before TTT match predictions after reset
    2. The reset works even after training on a different sequence
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return

    different_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHANVKSAVTRYNDDKLPGLRSFLLDAQT"  
    
    base_model = esm.pretrained.esmfold_v1()
    base_model = base_model.eval().cuda()

    ttt_cfg = DEFAULT_ESMFOLD_TTT_CFG
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    # ttt_cfg.initial_state_reset = True  # Ensure initial state is saved
    
    model = ESMFoldTTT.ttt_from_pretrained(base_model, ttt_cfg=ttt_cfg, esmfold_config=base_model.cfg)
    
    print(f"Making prediction on test sequence (length {len(TEST_SEQUENCE)}) before TTT")
    with torch.no_grad():
        tokens_before = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_before = model._ttt_predict_logits(tokens_before)
        pdb_str_before = model.infer_pdb(TEST_SEQUENCE)
        
    plddt_before = get_plddt_from_pdb(pdb_str_before)
    print(f"pLDDT before TTT: {plddt_before:.2f}")
    print(f"Applying TTT on different sequence (length {len(different_sequence)})")
    model.ttt(different_sequence)
    
    with torch.no_grad():
        tokens_after = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_after_ttt = model._ttt_predict_logits(tokens_after)
        pdb_str_after_ttt = model.infer_pdb(TEST_SEQUENCE)
    
    logits_diff = torch.abs(logits_before - logits_after_ttt).max().item()
    print(f"Max logits difference after TTT: {logits_diff:.6f}")

    plddt_after_ttt = get_plddt_from_pdb(pdb_str_after_ttt)
    print(f"pLDDT after TTT: {plddt_after_ttt:.2f}")
    
    if logits_diff < 1e-5:
        print("Warning: Model predictions didn't change much after TTT")
    else:
        print(f"Model predictions changed significantly after TTT")
    
    # print("Resetting model to initial state")
    model.ttt_reset()
    
    # print("Making prediction on test sequence after reset")
    with torch.no_grad():
        tokens_after = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_after = model._ttt_predict_logits(tokens_after)
        pdb_str_after_reset = model.infer_pdb(TEST_SEQUENCE)
    
    max_diff = torch.abs(logits_before - logits_after).max().item()
    mean_diff = torch.abs(logits_before - logits_after).mean().item()

    plddt_after_reset = get_plddt_from_pdb(pdb_str_after_reset)
    print(f"pLDDT after reset: {plddt_after_reset:.2f}")
    
    print(f"Max difference in logits after reset: {max_diff:.6e}")
    print(f"Mean difference in logits after reset: {mean_diff:.6e}")
    
    tolerance = 1e-5
    
    if max_diff < tolerance:
        print(f"TEST PASSED: Predictions match after reset (max diff: {max_diff:.6e} < {tolerance:.6e})")
    else:
        pred_tokens_before = logits_before.argmax(dim=-1)
        pred_tokens_after = logits_after.argmax(dim=-1)
        token_match_rate = (pred_tokens_before == pred_tokens_after).float().mean().item()
        
        raise AssertionError(
            f"TEST FAILED: Predictions don't match after reset.\n"
            f"Max difference: {max_diff:.6e} (tolerance: {tolerance:.6e})\n"
            f"Mean difference: {mean_diff:.6e}\n"
            f"Token match rate: {token_match_rate:.2%}\n"
            f"This suggests that ttt_reset() didn't properly restore the model state."
        )


if __name__ == "__main__":
    # test_esmfold_ttt_improves_plddt_2x()
    print("\n" + "="*80 + "\n")
    test_esmfold_ttt_reset()