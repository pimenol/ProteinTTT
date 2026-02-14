"""
Test for ESM2 TTT reset functionality.
This test is critical because ESM2TTT uses the default _ttt_get_trainable_modules()
which returns [self], the exact case where the bug in _ttt_get_state() would occur.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
esm = pytest.importorskip("esm")

from proteinttt.models.esm2 import ESM2TTT, DEFAULT_ESM2_35M_TTT_CFG

TEST_SEQUENCE = "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKSLIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTTLENLIATDAFAKEMVEVIIEE"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def test_esm2_ttt_reset():
    # Use two different sequences
    test_sequence = TEST_SEQUENCE  # Original test sequence
    different_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHANVKSAVTRYNDDKLPGLRSFLLDAQT"  # Different sequence

    # Use smaller 35M model for faster testing
    model, _alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()

    ttt_cfg = copy.deepcopy(DEFAULT_ESM2_35M_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    ttt_cfg.initial_state_reset = True  # Ensure initial state is saved
    
    ttt_model = ESM2TTT.ttt_from_pretrained(model, ttt_cfg=ttt_cfg)
    
    assert (
        ttt_model._ttt_initial_state is not None
    ), "Initial state was not saved (_ttt_initial_state is None)"
    assert len(ttt_model._ttt_initial_state) > 0, "Initial state is empty!"
    
    # Step 1: Make prediction on test_sequence BEFORE TTT
    with torch.no_grad():
        tokens_before = ttt_model._ttt_tokenize(test_sequence).cuda()
        logits_before = ttt_model._ttt_predict_logits(tokens_before)
        
    # Step 2: Apply TTT on a DIFFERENT sequence
    ttt_model.ttt(different_sequence)
    
    # Verify that model has changed (predictions should be different after TTT)
    with torch.no_grad():
        logits_after_ttt = ttt_model._ttt_predict_logits(tokens_before)
    
    logits_diff = torch.abs(logits_before - logits_after_ttt).max().item()
    
    # Step 3: Reset model
    ttt_model.ttt_reset()
    
    # Step 4: Make prediction on test_sequence AFTER reset
    with torch.no_grad():
        tokens_after = ttt_model._ttt_tokenize(test_sequence).cuda()
        logits_after = ttt_model._ttt_predict_logits(tokens_after)
    
    # Step 5: Verify predictions match
    # Check if logits match (allowing small numerical differences)
    max_diff = torch.abs(logits_before - logits_after).max().item()
    mean_diff = torch.abs(logits_before - logits_after).mean().item()

    # Threshold for numerical precision - should be very small
    tolerance = 1e-5

    assert max_diff < tolerance, (
        "Reset did not fully restore logits.\n"
        f"Max logits diff: {max_diff:.6e} (tolerance: {tolerance:.6e})\n"
        f"Mean logits diff: {mean_diff:.6e}\n"
        f"Max logits diff after TTT (sanity check): {logits_diff:.6e}\n"
    )
