"""
Test for ProteinTTT + ESMFold.
Verifies that pLDDT improves at least 2x after applying TTT.
"""

import copy
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
esm = pytest.importorskip("esm")
bsio = pytest.importorskip("biotite.structure.io")

from proteinttt.models.esmfold import ESMFoldTTT, DEFAULT_ESMFOLD_TTT_CFG

TEST_SEQUENCE = "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKSLIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTTLENLIATDAFAKEMVEVIIEE"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


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


@pytest.mark.slow
def test_esmfold_ttt_improves_plddt_2x():
    """
    Test that ProteinTTT improves pLDDT by at least 2x.
    """
    sequence = TEST_SEQUENCE

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    with torch.no_grad():
        baseline_output = model.infer_pdb(sequence)
    baseline_plddt = get_plddt_from_pdb(baseline_output)

    ttt_cfg = copy.deepcopy(DEFAULT_ESMFOLD_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10

    ttt_model = ESMFoldTTT.ttt_from_pretrained(model, ttt_cfg=ttt_cfg, esmfold_config=model.cfg)
    ttt_model.ttt(sequence)

    with torch.no_grad():
        ttt_output = ttt_model.infer_pdb(sequence)
    ttt_plddt = get_plddt_from_pdb(ttt_output)

    improvement_ratio = ttt_plddt / baseline_plddt
    ttt_model.ttt_reset()

    assert improvement_ratio >= 2.0, (
        f"pLDDT improvement was only {improvement_ratio:.2f}x "
        f"(baseline: {baseline_plddt:.2f}, after TTT: {ttt_plddt:.2f}); "
        "expected at least 2x improvement."
    )


def test_esmfold_ttt_reset():
    """
    Test that ttt_reset() properly restores the model to its initial state.
    
    This test verifies that:
    1. Model predictions before TTT match predictions after reset
    2. The reset works even after training on a different sequence
    """
    different_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHANVKSAVTRYNDDKLPGLRSFLLDAQT"  
    
    base_model = esm.pretrained.esmfold_v1()
    base_model = base_model.eval().cuda()

    ttt_cfg = copy.deepcopy(DEFAULT_ESMFOLD_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    # ttt_cfg.initial_state_reset = True  # Ensure initial state is saved
    
    model = ESMFoldTTT.ttt_from_pretrained(base_model, ttt_cfg=ttt_cfg, esmfold_config=base_model.cfg)
    
    assert len(model._ttt_initial_state) > 0, "Initial state is empty!"

    with torch.no_grad():
        tokens_before = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_before = model._ttt_predict_logits(tokens_before)
        pdb_str_before = model.infer_pdb(TEST_SEQUENCE)
        
    plddt_before = get_plddt_from_pdb(pdb_str_before)

    model.ttt(different_sequence)
    
    with torch.no_grad():
        tokens_after = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_after_ttt = model._ttt_predict_logits(tokens_after)
        pdb_str_after_ttt = model.infer_pdb(TEST_SEQUENCE)
    
    logits_diff = torch.abs(logits_before - logits_after_ttt).max().item()
    plddt_after_ttt = get_plddt_from_pdb(pdb_str_after_ttt)
    
    model.ttt_reset()
    
    with torch.no_grad():
        tokens_after = model._ttt_tokenize(TEST_SEQUENCE).cuda()
        logits_after = model._ttt_predict_logits(tokens_after)
        pdb_str_after_reset = model.infer_pdb(TEST_SEQUENCE)
    
    max_diff = torch.abs(logits_before - logits_after).max().item()
    mean_diff = torch.abs(logits_before - logits_after).mean().item()

    plddt_after_reset = get_plddt_from_pdb(pdb_str_after_reset)
    
    tolerance = 1e-5
    plddt_tolerance = 1e-3
    plddt_diff = float(abs(plddt_before - plddt_after_reset))

    assert max_diff < tolerance, (
        "Reset did not fully restore logits.\n"
        f"Max logits diff: {max_diff:.6e} (tolerance: {tolerance:.6e})\n"
        f"Mean logits diff: {mean_diff:.6e}\n"
        f"Max logits diff after TTT (sanity check): {logits_diff:.6e}\n"
    )
    assert plddt_diff < plddt_tolerance, (
        "Reset did not fully restore pLDDT.\n"
        f"pLDDT before: {plddt_before:.6f}\n"
        f"pLDDT after reset: {plddt_after_reset:.6f}\n"
        f"Abs diff: {plddt_diff:.6e} (tolerance: {plddt_tolerance:.6e})\n"
        f"pLDDT after TTT (sanity check): {plddt_after_ttt:.6f}\n"
    )
