# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProteinTTT is a framework for test-time training (TTT) of protein language models — customizing a pretrained model to one protein at a time for improved predictions. Paper: [arXiv:2411.02109](https://arxiv.org/abs/2411.02109).

## Commands

```bash
# Install (editable mode, after installing the target model like ESMFold separately)
pip install -e .

# Run tests (require CUDA GPU + pretrained models)
python -m pytest tests/
python tests/test_esmfold_ttt.py    # ESMFold integration tests
python tests/test_esm2_ttt_reset.py # ESM2 reset validation

# Format code
black --line-length 80 proteinttt/

# Run batch pipeline via SLURM
sbatch scripts/run_df.sh
# Or directly
python scripts/run_df.py --config scripts/config.yaml
```

## Architecture

### Core abstraction: `proteinttt/base.py`

`TTTConfig` — dataclass with all hyperparameters (LR, steps, LoRA, masking, FGR, etc.). Supports `TTTConfig.from_yaml()`.

`TTTModule(ABC, nn.Module)` — abstract base class (~1500 lines) implementing the entire TTT loop:
- `ttt(sequence, ...)` — main entry: tokenize → setup optimizer/LoRA → masked-LM training loop → return metrics
- `ttt_reset()` — restore model to pre-TTT state (deep-copied at init)
- Abstract methods that each model must implement: `_ttt_tokenize()`, `_ttt_predict_logits()`, `_ttt_get_trainable_modules()`, `_ttt_eval_step()`

### Model implementations: `proteinttt/models/`

Each file subclasses `TTTModule` alongside the original model class:

| Class | File | Base model | Notes |
|-------|------|------------|-------|
| `ESMFoldTTT` | `esmfold.py` | ESMFold | Structure prediction, evaluates pLDDT/TM-score |
| `ESM2TTT` | `esm2.py` | ESM2 | Language modeling only, unnormalized CE loss |
| `ESM2TTT_HF` | `esm2_hf.py` | HuggingFace EsmForMaskedLM | HF-compatible variant |
| `SaProtTTT_HF` | `saprot_hf.py` | Extends ESM2TTT_HF | Seq+structure tokens, masks seq only |
| `ProGen2TTT` | `progen2.py` | ProGen2 (autoregressive) | batch_size must be 1 |
| `DPLM2BitTTT` | `dplm2bit.py` | DPLM2 | Discrete diffusion PLM |
| `ProSSTTTT` | `prosst.py` | ProSST | Structure-aware |
| `MSATransformerTTT` | `msa_transformer.py` | MSA Transformer | MSA-based |

### Utilities: `proteinttt/utils/`

- `torch.py` — `preserve_model_state()` decorator, `get_optimal_window()` for sequence cropping
- `msa.py` — A3M parsing, `MSAServer` class for caching/building MSAs via MMseqs2
- `structure.py` — TM-score (via TMalign) and lDDT calculations
- `io.py` — logger setup
- `plots.py` — metrics-vs-step visualization

### Key data flow in `ttt()`

1. Tokenize input sequence (and optional MSA)
2. Freeze all params → apply LoRA (optional) → unfreeze target modules
3. For each step: sample batch → BERT-mask tokens → forward → loss → backward → optimizer step
4. Periodically: score sequence (pseudo-perplexity), run eval, compute FGR metrics
5. Optionally reset to best-confidence state

### FGR (Fidelity-Gain Ratio) — `feature/fgr` branch

Monitors semantic drift during TTT by comparing current representation to initial anchor (`z_0`). Computes `loss_delta / drift_delta` ratio for early stopping decisions. Config: `fgr_enabled`, `fgr_drift_threshold`, `fgr_ratio_threshold`, etc.

## Style Conventions

- Black formatter, 80 char line length, Python 3.8+ target
- Match existing patterns when adding new model implementations
- New models: subclass `TTTModule`, implement abstract methods, follow `esmfold.py` as reference

## Behavioral Guidelines

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- Minimum code that solves the problem. No speculative features or abstractions.
- Surgical changes only — touch only what's needed, match existing style.
- Define verifiable success criteria before implementing.
- Every changed line should trace directly to the user's request.
