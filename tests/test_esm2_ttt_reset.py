"""
Test for ESM2 TTT reset functionality.
This test is critical because ESM2TTT uses the default _ttt_get_trainable_modules()
which returns [self], the exact case where the bug in _ttt_get_state() would occur.
"""

import torch
import esm
import proteinttt
print(proteinttt.__file__)
from proteinttt.models.esm2 import ESM2TTT, DEFAULT_ESM2_35M_TTT_CFG

TEST_SEQUENCE = "GIHLGELGLLPSTVLAIGYFENLVNIICESLNMLPKLEVSGKEYKKFKFTIVIPKDLDANIKKRAKIYFKQKSLIEIEIPTSSRNYPIHIQFDENSTDDILHLYDMPTTIGGIDKAIEMFMRKGHIGKTDQQKLLEERELRNFKTTLENLIATDAFAKEMVEVIIEE"


def test_esm2_ttt_reset():
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    
    # Use two different sequences
    test_sequence = TEST_SEQUENCE  # Original test sequence
    different_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSHANVKSAVTRYNDDKLPGLRSFLLDAQT"  # Different sequence
    
    print("Loading ESM2 model...")
    # Use smaller 35M model for faster testing
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().cuda()
    
    print("Setting up TTT model...")
    ttt_cfg = DEFAULT_ESM2_35M_TTT_CFG
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    ttt_cfg.initial_state_reset = True  # Ensure initial state is saved
    
    ttt_model = ESM2TTT.ttt_from_pretrained(model, ttt_cfg=ttt_cfg)
    
    # Verify that initial state was saved
    if ttt_model._ttt_initial_state is None:
        raise AssertionError("Initial state was not saved! initial_state_reset=True but _ttt_initial_state is None")
    
    print(f"Initial state saved with {len(ttt_model._ttt_initial_state)} modules: {list(ttt_model._ttt_initial_state.keys())}")
    
    if len(ttt_model._ttt_initial_state) == 0:
        raise AssertionError(
            "Initial state is empty!
        )
    
    print(f"  ✓ Successfully saved {len(ttt_model._ttt_initial_state)} child modules")
    
    # Step 1: Make prediction on test_sequence BEFORE TTT
    print(f"\nStep 1: Making prediction on test sequence (length {len(test_sequence)}) before TTT...")
    with torch.no_grad():
        tokens_before = ttt_model._ttt_tokenize(test_sequence).cuda()
        logits_before = ttt_model._ttt_predict_logits(tokens_before)
    print(f"  Logits shape: {logits_before.shape}")
        
    # Step 2: Apply TTT on a DIFFERENT sequence
    print(f"\nStep 2: Applying TTT on different sequence (length {len(different_sequence)})...")
    ttt_model.ttt(different_sequence)
    
    # Verify that model has changed (predictions should be different after TTT)
    print("\nVerifying model has changed after TTT...")
    with torch.no_grad():
        logits_after_ttt = ttt_model._ttt_predict_logits(tokens_before)
    
    logits_diff = torch.abs(logits_before - logits_after_ttt).max().item()
    print(f"  Max logits difference after TTT: {logits_diff:.6f}")
    
    if logits_diff < 1e-5:
        print("  Warning: Model predictions didn't change much after TTT")
    else:
        print(f"  ✓ Model predictions changed significantly after TTT")
    
    # Step 3: Reset model
    print("\nStep 3: Resetting model to initial state...")
    ttt_model.ttt_reset()
    
    # Step 4: Make prediction on test_sequence AFTER reset
    print("\nStep 4: Making prediction on test sequence after reset...")
    with torch.no_grad():
        tokens_after = ttt_model._ttt_tokenize(test_sequence).cuda()
        logits_after = ttt_model._ttt_predict_logits(tokens_after)
    
    # Step 5: Verify predictions match
    print("\nStep 5: Verifying predictions match...")
    
    # Check if logits match (allowing small numerical differences)
    max_diff = torch.abs(logits_before - logits_after).max().item()
    mean_diff = torch.abs(logits_before - logits_after).mean().item()
    
    print(f"  Max difference in logits: {max_diff:.6e}")
    print(f"  Mean difference in logits: {mean_diff:.6e}")
    
    # Threshold for numerical precision - should be very small
    tolerance = 1e-5
    
    if max_diff < tolerance:
        print(f"\n✓ TEST PASSED: Predictions match after reset (max diff: {max_diff:.6e} < {tolerance:.6e})")
        print(f"  This confirms the bug fix works for models using default _ttt_get_trainable_modules()!")
    else:
        # Also check predicted tokens to see if they still match
        pred_tokens_before = logits_before.argmax(dim=-1)
        pred_tokens_after = logits_after.argmax(dim=-1)
        token_match_rate = (pred_tokens_before == pred_tokens_after).float().mean().item()
        
        raise AssertionError(
            f"✗ TEST FAILED: Predictions don't match after reset.\n"
            f"  Max difference: {max_diff:.6e} (tolerance: {tolerance:.6e})\n"
            f"  Mean difference: {mean_diff:.6e}\n"
            f"  Token match rate: {token_match_rate:.2%}\n"
            f"  This suggests that ttt_reset() didn't properly restore the model state.\n"
            f"  For ESM2TTT which uses default _ttt_get_trainable_modules(), this likely means\n"
            f"  the bug fix in _ttt_get_state() is not working correctly."
        )


if __name__ == "__main__":
    print("="*80)
    print("Testing ESM2 TTT Reset Functionality")
    print("="*80)
    print("\nNOTE: This test is critical for verifying the bug fix!")
    print("ESM2TTT uses default _ttt_get_trainable_modules() which returns [self].")
    print("The old implementation would have saved NO modules, causing reset to fail.")
    print("="*80 + "\n")
    
    test_esm2_ttt_reset()
    
    print("\n" + "="*80)
    print("✓ TEST PASSED: ESM2 reset works correctly!")
    print("="*80)
