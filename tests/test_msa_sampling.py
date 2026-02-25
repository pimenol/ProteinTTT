"""
Tests for MSA sampling strategies: random, top, neighbors, cluster.

Unit tests (ClusterMSA, homology weights) run without GPU.
Integration tests (_ttt_sample_batch with each strategy) require CUDA.
"""

import copy
import time
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests — no GPU required
# ---------------------------------------------------------------------------


class TestClusterMSA:
    """Test proteinttt.utils.ClusterMSA.cluster_msa."""

    def test_returns_correct_shape(self):
        from proteinttt.utils.ClusterMSA import cluster_msa

        rng = np.random.default_rng(42)
        tokens = rng.integers(0, 20, size=(30, 20)).astype(np.uint8)
        labels = cluster_msa(tokens, eps=10.0)
        assert labels.shape == (30,)

    def test_auto_eps_scan(self):
        from proteinttt.utils.ClusterMSA import cluster_msa

        rng = np.random.default_rng(42)
        tokens = rng.integers(0, 20, size=(30, 20)).astype(np.uint8)
        labels = cluster_msa(tokens)  # eps=None triggers scan
        assert labels.shape == (30,)

    def test_finds_obvious_clusters(self):
        """Three well-separated groups should produce >= 2 clusters."""
        from proteinttt.utils.ClusterMSA import cluster_msa

        rng = np.random.default_rng(0)
        seq_len = 30
        n_per_group = 20
        # Build three tight groups: each member is a centroid with a few mutations.
        # Centroids use non-overlapping tokens so between-group one-hot distance is large.
        groups = []
        for base_tok in (0, 10, 19):
            centroid = np.full((1, seq_len), base_tok, dtype=np.uint8)
            block = np.tile(centroid, (n_per_group, 1))
            # Mutate ~3 positions per sequence to a nearby token
            for i in range(n_per_group):
                positions = rng.choice(seq_len, size=3, replace=False)
                block[i, positions] = np.clip(base_tok + rng.integers(-1, 2, size=3), 0, 19).astype(np.uint8)
            groups.append(block)
        tokens = np.vstack(groups)

        labels = cluster_msa(tokens, eps=5.0, min_samples=2)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 2, f"Expected >= 2 clusters, got {n_clusters}"

    def test_small_msa_does_not_crash(self):
        """Fewer sequences than min_samples should still return labels."""
        from proteinttt.utils.ClusterMSA import cluster_msa

        tokens = np.zeros((2, 10), dtype=np.uint8)
        labels = cluster_msa(tokens, eps=5.0, min_samples=3)
        assert labels.shape == (2,)


class TestHomologyWeights:
    """Test proteinttt.utils.sampling.compute_homology_weights."""

    def test_returns_valid_probability_distribution(self):
        from proteinttt.utils.sampling import compute_homology_weights

        rng = np.random.default_rng(42)
        msa = rng.integers(0, 20, size=(10, 30)).astype(np.uint8)
        n_eff, weights = compute_homology_weights(
            ungapped_msa=msa, gap_token=20, can_use_torch=False,
        )
        assert isinstance(n_eff, (int, float, np.floating))
        assert n_eff > 0
        assert weights.shape == (10,)
        assert np.all(weights >= 0)
        assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    def test_identical_seqs_get_equal_weights(self):
        from proteinttt.utils.sampling import compute_homology_weights

        seq = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        msa = np.tile(seq, (5, 1))
        _, weights = compute_homology_weights(
            ungapped_msa=msa, gap_token=20, can_use_torch=False,
        )
        # All identical → same neighbor count → equal weights
        assert np.allclose(weights, weights[0])


# ---------------------------------------------------------------------------
# Integration tests — require GPU + ESM model
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
esm = pytest.importorskip("esm")

from proteinttt.models.esm2 import ESM2TTT, DEFAULT_ESM2_35M_TTT_CFG


def _needs_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )


@pytest.fixture(scope="module")
def esm2_base_model():
    """Load a small ESM2 model once for the whole module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    return model.eval().cuda()


def _make_ttt_model(base_model, strategy, batch_size=4):
    """Create ESM2TTT with the given MSA sampling strategy."""
    cfg = copy.deepcopy(DEFAULT_ESM2_35M_TTT_CFG)
    cfg.msa_sampling_strategy = strategy
    cfg.seed = 0
    cfg.steps = 2
    cfg.ags = 1
    cfg.batch_size = batch_size
    cfg.msa = True
    return ESM2TTT.ttt_from_pretrained(base_model, ttt_cfg=cfg)


def _make_synthetic_msa(model, n_seqs=20, seed=42):
    """Build a synthetic tokenized MSA by mutating a short sequence."""
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 AA
    base_tokens = model._ttt_tokenize(seq)  # [1, seq_len]

    msa = base_tokens.repeat(n_seqs, 1)
    non_special = model._ttt_get_non_special_tokens()
    rng = np.random.default_rng(seed)
    # Mutate rows 1..n_seqs-1 (row 0 stays as the query)
    for i in range(1, n_seqs):
        n_mut = rng.integers(1, 10)
        # Avoid special-token positions (first=BOS, last=EOS)
        positions = rng.choice(
            range(1, msa.shape[1] - 1), size=n_mut, replace=False,
        )
        for pos in positions:
            msa[i, pos] = non_special[rng.integers(0, len(non_special))]
    return msa


def _setup_strategy_data(model, msa, strategy):
    """Pre-compute sampling data the same way ttt() would."""
    model._msa_sampling_weights = None
    model._msa_cluster_labels = None

    if strategy == "neighbors":
        from proteinttt.utils.sampling import compute_homology_weights

        msa_np = msa.cpu().numpy().astype(np.uint8)
        gap_token = model._ttt_get_padding_token()
        _, weights = compute_homology_weights(
            ungapped_msa=msa_np, gap_token=gap_token, can_use_torch=False,
        )
        model._msa_sampling_weights = torch.from_numpy(weights).float()

    elif strategy == "cluster":
        from proteinttt.utils.ClusterMSA import cluster_msa

        msa_np = msa.cpu().numpy().astype(np.uint8)
        model._msa_cluster_labels = cluster_msa(msa_np, eps=10.0)


# ---- Parametrized test: basic shapes are correct for every strategy -------

@_needs_cuda()
@pytest.mark.parametrize("strategy", ["random", "top", "neighbors", "cluster"])
def test_sample_batch_shapes(esm2_base_model, strategy):
    """_ttt_sample_batch returns tensors with the correct shapes."""
    model = _make_ttt_model(esm2_base_model, strategy)
    msa = _make_synthetic_msa(model)
    _setup_strategy_data(model, msa, strategy)

    batch_masked, targets, mask, start_indices = model._ttt_sample_batch(msa)

    bs = model.ttt_cfg.batch_size
    crop = min(model.ttt_cfg.crop_size, msa.shape[1])

    assert batch_masked.shape == (bs, crop)
    assert targets.shape == (bs, crop)
    assert mask.shape == (bs, crop)
    assert mask.dtype == torch.bool
    assert start_indices.shape == (bs,)
    assert mask.any(), "At least some tokens should be masked"


# ---- Strategy-specific behavioural tests ----------------------------------

@_needs_cuda()
def test_top_always_selects_first_sequences(esm2_base_model):
    """With 'top', the batch should always contain the first sequences."""
    model = _make_ttt_model(esm2_base_model, "top", batch_size=4)
    msa = _make_synthetic_msa(model, n_seqs=20)
    _setup_strategy_data(model, msa, "top")

    # Call multiple times – top should always pick indices 0..3
    for _ in range(5):
        _, targets, _, _ = model._ttt_sample_batch(msa)
        # targets (before masking) should match the first 4 rows
        crop = min(model.ttt_cfg.crop_size, msa.shape[1])
        expected = msa[:4, :crop]
        assert torch.equal(targets, expected), (
            "top strategy should deterministically select the first batch_size sequences"
        )


@_needs_cuda()
def test_random_varies_across_calls(esm2_base_model):
    """With 'random', repeated calls should (almost certainly) give different batches."""
    model = _make_ttt_model(esm2_base_model, "random", batch_size=4)
    msa = _make_synthetic_msa(model, n_seqs=20)
    _setup_strategy_data(model, msa, "random")

    targets_list = []
    for _ in range(5):
        _, targets, _, _ = model._ttt_sample_batch(msa)
        targets_list.append(targets.clone())

    # At least one pair of calls should differ
    all_same = all(torch.equal(targets_list[0], t) for t in targets_list[1:])
    assert not all_same, "random strategy should produce varying batches"


@_needs_cuda()
def test_neighbors_respects_weights(esm2_base_model):
    """With 'neighbors', sequences with higher weight should appear more often."""
    model = _make_ttt_model(esm2_base_model, "neighbors", batch_size=4)
    msa = _make_synthetic_msa(model, n_seqs=20)

    # Create artificial weights that heavily favour one sequence
    n = msa.shape[0]
    weights = torch.ones(n) * 1e-6
    weights[0] = 1.0  # almost all weight on row 0
    weights /= weights.sum()
    model._msa_sampling_weights = weights
    model._msa_cluster_labels = None

    crop = min(model.ttt_cfg.crop_size, msa.shape[1])
    row0 = msa[0, :crop]

    # With replacement=False and batch_size=4, row 0 can appear at most once
    # per batch.  With weight ≈ 1.0 it should be selected in *every* batch.
    batches_with_row0 = 0
    n_calls = 20
    for _ in range(n_calls):
        _, targets, _, _ = model._ttt_sample_batch(msa)
        for b in range(targets.shape[0]):
            if torch.equal(targets[b], row0):
                batches_with_row0 += 1
                break  # found in this batch, move on

    ratio = batches_with_row0 / n_calls
    assert ratio > 0.8, (
        f"neighbors: expected row 0 in most batches (ratio={ratio:.2f})"
    )


@_needs_cuda()
def test_cluster_samples_from_multiple_clusters(esm2_base_model):
    """With 'cluster', a batch should draw from more than one cluster."""
    model = _make_ttt_model(esm2_base_model, "cluster", batch_size=6)
    msa = _make_synthetic_msa(model, n_seqs=30)

    # Assign artificial cluster labels: 3 clusters of 10
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    model._msa_cluster_labels = labels
    model._msa_sampling_weights = None

    # Collect which clusters are represented across several calls
    cluster_hits = set()
    for _ in range(10):
        indices = model._ttt_cluster_sample_indices(labels, batch_size=6)
        for idx in indices.tolist():
            cluster_hits.add(labels[idx])

    assert len(cluster_hits) >= 2, (
        f"cluster strategy should sample from multiple clusters, got {cluster_hits}"
    )


@_needs_cuda()
def test_cluster_fallback_when_no_clusters(esm2_base_model):
    """If DBSCAN finds no clusters, 'cluster' should fall back to random."""
    model = _make_ttt_model(esm2_base_model, "cluster", batch_size=4)
    msa = _make_synthetic_msa(model, n_seqs=20)

    # All unclustered
    model._msa_cluster_labels = np.full(20, -1, dtype=int)
    model._msa_sampling_weights = None

    # Should not crash and should still return valid shapes
    batch_masked, targets, mask, start_indices = model._ttt_sample_batch(msa)
    bs = model.ttt_cfg.batch_size
    crop = min(model.ttt_cfg.crop_size, msa.shape[1])
    assert batch_masked.shape == (bs, crop)


@_needs_cuda()
def test_strategies_with_small_msa(esm2_base_model):
    """All strategies should handle MSA with fewer sequences than batch_size."""
    for strategy in ["random", "top", "neighbors", "cluster"]:
        model = _make_ttt_model(esm2_base_model, strategy, batch_size=8)
        msa = _make_synthetic_msa(model, n_seqs=3)  # fewer than batch_size
        _setup_strategy_data(model, msa, strategy)

        batch_masked, targets, mask, start_indices = model._ttt_sample_batch(msa)
        crop = min(model.ttt_cfg.crop_size, msa.shape[1])
        assert batch_masked.shape == (8, crop), (
            f"strategy={strategy} failed with small MSA"
        )


@_needs_cuda()
def test_strategies_with_single_sequence(esm2_base_model):
    """All strategies should handle single-sequence input (no MSA)."""
    for strategy in ["random", "top", "neighbors", "cluster"]:
        model = _make_ttt_model(esm2_base_model, strategy, batch_size=4)
        seq = "ACDEFGHIKLMNPQRSTVWY"
        x = model._ttt_tokenize(seq)  # [1, seq_len]

        # With a single sequence, no pre-computed data is needed
        model._msa_sampling_weights = None
        model._msa_cluster_labels = None

        batch_masked, targets, mask, start_indices = model._ttt_sample_batch(x)
        crop = min(model.ttt_cfg.crop_size, x.shape[1])
        assert batch_masked.shape == (4, crop), (
            f"strategy={strategy} failed with single sequence"
        )


# ---------------------------------------------------------------------------
# End-to-end integration test — real MSA + full ttt() loop + timing
# ---------------------------------------------------------------------------

# MSA file: A0A6J5KJJ6.a3m (71 sequences, 153 AA query — median-sized MSA
# from /scratch/project/open-35-8/antonb/bfvd/bfvd_msa)
_MSA_FILE = Path(__file__).parent / "A0A6J5KJJ6.a3m"
_QUERY_SEQ = (
    "MTEEDKPKHKPPARAGRPVKGAGVKVRYPYLAVDLQMPVVYRDQLFLPGGQRAIPSNRGK"
    "GEVKVDFWSSHCAECGVLFSFYTVPRDEEKPFGFLRRCETHRQKGKPIDLAKFRATRPPF"
    "LEVAEQIEQGKRMWGLMWGMWFDQDAAKLARGG"
)

STRATEGIES = ["random", "top", "neighbors", "cluster"]


def _make_ttt_model_for_e2e(base_model, strategy):
    """Create ESM2TTT tuned for a short end-to-end run with real MSA."""
    cfg = copy.deepcopy(DEFAULT_ESM2_35M_TTT_CFG)
    cfg.msa_sampling_strategy = strategy
    cfg.seed = 0
    cfg.steps = 3           # few steps, enough to verify the loop runs
    cfg.ags = 1
    cfg.batch_size = 4
    cfg.msa = True
    cfg.initial_state_reset = True
    cfg.score_seq_kind = None      # skip scoring for speed
    cfg.eval_each_step = False
    return ESM2TTT.ttt_from_pretrained(base_model, ttt_cfg=cfg)


@_needs_cuda()
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_ttt_e2e_with_real_msa(esm2_base_model, strategy):
    """Run ttt() end-to-end on a real A3M MSA for each sampling strategy.

    Checks:
      - ttt() completes without error
      - Returned dict contains expected keys (df, ttt_step_data)
      - Loss values are finite numbers
    """
    assert _MSA_FILE.exists(), f"Test MSA file not found: {_MSA_FILE}"

    model = _make_ttt_model_for_e2e(esm2_base_model, strategy)
    result = model.ttt(seq=_QUERY_SEQ, msa_pth=_MSA_FILE)

    # Check return structure
    assert "df" in result, "ttt() should return a dict with 'df'"
    assert "ttt_step_data" in result, "ttt() should return 'ttt_step_data'"

    df = result["df"]
    assert len(df) > 0, "df should have at least one logged step"
    assert "loss" in df.columns, "df should contain 'loss' column"

    # All loss values should be finite
    losses = df["loss"].dropna().values
    assert len(losses) > 0, "Should have at least one loss value"
    for l in losses:
        assert np.isfinite(l), f"Non-finite loss detected: {l}"

    # Reset model for clean state
    model.ttt_reset()


@_needs_cuda()
def test_ttt_e2e_timing_comparison(esm2_base_model):
    """Run ttt() with all sampling strategies on a real MSA and compare wall-clock time.

    This test prints a timing summary so users can compare strategy overhead.
    It does NOT assert one is faster than another (hardware-dependent), but
    verifies all strategies complete and reports times.
    """
    assert _MSA_FILE.exists(), f"Test MSA file not found: {_MSA_FILE}"

    timing: dict[str, float] = {}

    for strategy in STRATEGIES:
        model = _make_ttt_model_for_e2e(esm2_base_model, strategy)

        # Warm-up CUDA
        if strategy == STRATEGIES[0]:
            _dummy = torch.randn(1, device="cuda")
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        result = model.ttt(seq=_QUERY_SEQ, msa_pth=_MSA_FILE)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        timing[strategy] = elapsed

        # Basic sanity — loss should decrease or at least not blow up
        df = result["df"]
        losses = df["loss"].dropna().values
        assert len(losses) > 0
        assert all(np.isfinite(losses)), f"Non-finite loss with strategy={strategy}"

        model.ttt_reset()
        del model

    # ---- Print timing summary ----
    print("\n" + "=" * 60)
    print("TTT end-to-end timing comparison (real MSA: A0A6J5KJJ6.a3m)")
    print(f"  MSA: 71 sequences, 153 AA query, 3 steps, batch_size=4")
    print("-" * 60)
    for strat, t in timing.items():
        print(f"  {strat:<12s}  {t:8.3f} s")
    print("=" * 60 + "\n")

    # All strategies should finish (no hang / crash)
    assert len(timing) == len(STRATEGIES)
