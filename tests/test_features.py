"""Feature-pipeline tests.

The golden test here is the most important one in the suite. `build_state_sequence`
was rewritten from a per-window Python loop into vectorized aggregation, and the
shipped checkpoints' normalization statistics were fit on the *old* values. Any
numerical drift would silently corrupt every prediction the model makes while
leaving all the plumbing looking healthy, so the output is locked bit-for-bit
against a stored fixture captured from the original implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netforecast.features import (
    N_FEATURES,
    STATE_FEATURE_NAMES,
    build_state_sequence,
    compute_window_state,
    missing_columns,
    window_label,
)

from .conftest import FIXTURES


@pytest.mark.skipif(not (FIXTURES / "golden_synthetic_w30.npz").exists(),
                    reason="golden fixture not present")
def test_vectorized_matches_golden_bit_for_bit(synthetic_df):
    """Locks the vectorized rewrite to the original loop implementation."""
    g = np.load(FIXTURES / "golden_synthetic_w30.npz")
    states, label_ids, timestamps, widx = build_state_sequence(synthetic_df, 30, with_index=True)

    assert np.array_equal(states, g["states"]), "state matrix drifted from the golden fixture"
    assert np.array_equal(label_ids, g["label_ids"])
    assert np.array_equal(timestamps, g["timestamps"])

    # window index must reproduce the old list-of-lists exactly
    lens = g["flow_lens"]
    assert np.array_equal(widx.ends - widx.starts, lens)
    flat, off = g["flow_flat"], 0
    for i in range(len(lens)):
        assert np.array_equal(widx.rows_for(i), flat[off:off + lens[i]])
        off += lens[i]


def test_vectorized_matches_reference_loop(synthetic_df):
    """Independently re-derive a sample of windows with the reference function."""
    df = synthetic_df
    states, label_ids, _, widx = build_state_sequence(df, 30, with_index=True)

    rng = np.random.default_rng(0)
    nonempty = [i for i in range(len(states)) if widx.count_for(i) > 0]
    for i in rng.choice(nonempty, size=min(40, len(nonempty)), replace=False):
        win = df.iloc[widx.rows_for(int(i))]
        assert np.allclose(compute_window_state(win), states[int(i)], rtol=1e-5, atol=1e-6), \
            f"window {i} disagrees with the reference implementation"
        assert window_label(win) != "" and label_ids[int(i)] >= 0


def test_shapes_and_names():
    assert len(STATE_FEATURE_NAMES) == N_FEATURES == 33
    assert len(set(STATE_FEATURE_NAMES)) == N_FEATURES, "feature names must be unique"


def test_empty_window_is_zero_state():
    assert np.array_equal(compute_window_state(pd.DataFrame()),
                          np.zeros(N_FEATURES, dtype=np.float32))


def test_gaps_produce_zero_state_windows(minimal_flows):
    """A quiet interval must still advance the sequence, not be skipped."""
    df = minimal_flows.copy()
    df.loc[df.index[-1], "timestamp"] = 500        # leave a long gap
    states, _, timestamps, _ = build_state_sequence(df, 30)
    assert len(states) == (500 // 30) + 1
    assert np.array_equal(timestamps, np.arange(len(states)) * 30)
    assert not states[5].any(), "a window with no flows should be the zero state"


def test_missing_columns_are_reported_not_crashed(minimal_flows):
    bad = minimal_flows.drop(columns=["ttl_mean", "dst_port"])
    assert set(missing_columns(bad)) == {"ttl_mean", "dst_port"}
    with pytest.raises(KeyError) as e:
        build_state_sequence(bad, 30)
    assert "ttl_mean" in str(e.value) and "dst_port" in str(e.value)


def test_empty_frame_rejected(minimal_flows):
    with pytest.raises(ValueError):
        build_state_sequence(minimal_flows.iloc[0:0], 30)


def test_nan_timestamp_rejected(minimal_flows):
    df = minimal_flows.copy()
    df.loc[df.index[0], "timestamp"] = np.nan
    with pytest.raises(ValueError, match="timestamp"):
        build_state_sequence(df, 30)


def test_no_nan_or_inf_in_output(synthetic_df):
    states, _, _, _ = build_state_sequence(synthetic_df, 30)
    assert np.isfinite(states).all()


def test_window_index_covers_every_row(synthetic_df):
    _, _, _, widx = build_state_sequence(synthetic_df, 30, with_index=True)
    assert int((widx.ends - widx.starts).sum()) == len(synthetic_df)
    assert sorted(widx.order.tolist()) == list(range(len(synthetic_df)))
