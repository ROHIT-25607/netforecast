"""Inference tests: batched paths must agree with the per-sample reference."""
from __future__ import annotations

import numpy as np
import pytest

from netforecast.features import N_FEATURES, build_state_sequence
from netforecast.mitre_mapping import MITRE_TACTIC, STAGES


@pytest.fixture(scope="module")
def states_norm(predictor, synthetic_df):
    states, _, _, _ = build_state_sequence(synthetic_df, predictor.cfg["window_seconds"])
    return predictor.normalize(states).astype(np.float32)


def test_rollout_shape_and_ranges(predictor, states_norm):
    L, k = predictor.cfg["context_len"], 5
    r = predictor.rollout(states_norm[-L:], k=k)

    assert len(r["forecast"]) == k
    for i, s in enumerate(r["forecast"], start=1):
        assert s["step"] == i
        assert 0.0 <= s["infiltration_probability"] <= 1.0
        assert 0.0 <= s["stage_confidence"] <= 1.0
        assert s["predicted_stage"] in STAGES
        assert s["mitre_tactic"] == MITRE_TACTIC[s["predicted_stage"]]

    assert len(r["attention_weights"]) == L
    assert r["attention_weights"] == pytest.approx(r["attention_weights"])
    assert sum(r["attention_weights"]) == pytest.approx(1.0, abs=1e-5), "attention must be a distribution"
    assert all(w >= 0 for w in r["attention_weights"])
    assert r["overall_infiltration_risk"] == pytest.approx(
        max(s["infiltration_probability"] for s in r["forecast"]))


def test_score_batch_matches_single(predictor, states_norm):
    """The batched scan must equal the per-anchor loop it replaced."""
    L = predictor.cfg["context_len"]
    anchors = list(range(L - 1, L - 1 + 60))
    single = [predictor.rollout(states_norm[t - L + 1: t + 1], k=1,
                                explain=False)["forecast"][0]["infiltration_probability"]
              for t in anchors]
    batched = predictor.score_batch(predictor.make_contexts(states_norm, anchors))
    assert np.allclose(batched["infiltration_probability"], single, atol=1e-5)


def test_rollout_batch_matches_single(predictor, states_norm):
    L, k = predictor.cfg["context_len"], 5
    anchors = [L - 1, L + 40, L + 120]
    rb = predictor.rollout_batch(predictor.make_contexts(states_norm, anchors), k=k)
    for i, t in enumerate(anchors):
        one = predictor.rollout(states_norm[t - L + 1: t + 1], k=k, explain=False)
        assert np.allclose(rb["infiltration_probability"][i],
                           [s["infiltration_probability"] for s in one["forecast"]], atol=1e-5)


def test_short_context_is_padded_not_crashed(predictor, states_norm):
    """Fewer than L observed windows must still produce a forecast."""
    r = predictor.rollout(states_norm[:2], k=3)
    assert len(r["forecast"]) == 3
    assert len(r["attention_weights"]) == predictor.cfg["context_len"]


def test_saliency_ranks_real_features(predictor, states_norm):
    from netforecast.features import STATE_FEATURE_NAMES
    r = predictor.rollout(states_norm[-predictor.cfg["context_len"]:], k=1, explain=True)
    top = r["top_driving_features"]
    assert 0 < len(top) <= 8
    names = [n for n, _ in top]
    assert all(n in STATE_FEATURE_NAMES for n in names)
    assert len(set(names)) == len(names)
    scores = [v for _, v in top]
    assert scores == sorted(scores, reverse=True), "saliency must be ranked descending"


def test_make_contexts_shape(predictor, states_norm):
    L = predictor.cfg["context_len"]
    ctx = predictor.make_contexts(states_norm, [0, 3, L - 1, L + 50])
    assert ctx.shape == (4, L, N_FEATURES)
    # anchors before a full context exists are left-padded, never truncated
    assert np.allclose(ctx[0][0], ctx[0][-1])


def test_normalize_roundtrip(predictor, states_norm):
    assert np.allclose(predictor.normalize(predictor.denormalize(states_norm)),
                       states_norm, atol=1e-4)
