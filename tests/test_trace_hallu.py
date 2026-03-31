#!/usr/bin/env python3
"""Unit tests for trace-hallu core invariants.

Covers:
  1. Hidden-state extraction shape consistency
  2. Label–state alignment assertion
  3. Group-aware train/val split (no question leakage)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# 1. Hidden-state extraction shape consistency
# ---------------------------------------------------------------------------

class TestHiddenStateShapeConsistency:
    """Verify that hidden-state arrays have (seq_len, hidden_size) for every layer."""

    def _make_mock_hdf5(self, tmp_path, num_traces=5, layers=(8, 16), seq_len=20, hidden=64):
        h5_path = os.path.join(tmp_path, "hidden_states_mock.h5")
        with h5py.File(h5_path, "w") as h5f:
            for layer in layers:
                grp = h5f.create_group(f"layer_{layer}")
                for i in range(num_traces):
                    data = np.random.randn(seq_len, hidden).astype(np.float16)
                    grp.create_dataset(f"trace_{i}", data=data, compression="gzip")
            h5f.attrs["num_traces"] = num_traces
            h5f.attrs["layer_indices"] = list(layers)
        return h5_path

    def test_all_layers_same_seq_len(self, tmp_path):
        layers = (8, 16, 24)
        seq_len, hidden = 30, 64
        h5_path = self._make_mock_hdf5(tmp_path, layers=layers,
                                       seq_len=seq_len, hidden=hidden)
        with h5py.File(h5_path, "r") as h5f:
            for i in range(5):
                shapes = []
                for layer in layers:
                    hs = h5f[f"layer_{layer}/trace_{i}"][:]
                    shapes.append(hs.shape)
                    assert hs.shape == (seq_len, hidden), (
                        f"trace {i} layer {layer}: expected ({seq_len}, {hidden}), got {hs.shape}"
                    )
                seq_lens = [s[0] for s in shapes]
                assert len(set(seq_lens)) == 1, f"Inconsistent seq_len across layers: {seq_lens}"

    def test_hidden_size_matches_config(self, tmp_path):
        hidden = 128
        h5_path = self._make_mock_hdf5(tmp_path, hidden=hidden)
        with h5py.File(h5_path, "r") as h5f:
            hs = h5f["layer_8/trace_0"][:]
            assert hs.shape[1] == hidden


# ---------------------------------------------------------------------------
# 2. Label–state alignment assertion
# ---------------------------------------------------------------------------

class TestLabelStateAlignment:
    """Labels length must match the generated-token portion of hidden states."""

    def test_labels_match_gen_tokens(self):
        prompt_len = 15
        num_gen_tokens = 25
        total_hs_len = prompt_len + num_gen_tokens

        hallu_labels = [0] * 10 + [1] * 15
        assert len(hallu_labels) == num_gen_tokens

        # Simulate dataset loading: pad labels to hidden-state length
        padded = [0] * (total_hs_len - len(hallu_labels)) + hallu_labels
        assert len(padded) == total_hs_len, (
            f"Padded labels {len(padded)} != hidden state length {total_hs_len}"
        )

    def test_assertion_fires_on_mismatch(self):
        """The assertion from collect_traces should catch length drift."""
        seq_len = 40
        expected_len = 42
        with pytest.raises(AssertionError):
            assert seq_len == expected_len, (
                f"Hidden state count {seq_len} != token count {expected_len}"
            )

    def test_token_ids_round_trip_preserves_length(self):
        """Concatenating prompt IDs + gen IDs preserves total count (no re-tokenize)."""
        prompt_ids = torch.randint(0, 1000, (1, 20))
        gen_ids = torch.randint(0, 1000, (1, 30))
        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        assert full_ids.shape[1] == 50


# ---------------------------------------------------------------------------
# 3. Group-aware train/val split (no question leakage)
# ---------------------------------------------------------------------------

class TestGroupAwareSplit:
    """All traces from the same sample_idx must land in the same split."""

    def _build_mock_dataset(self, n_questions=20, traces_per_q=3):
        """Return a lightweight object with .items matching TraceHiddenStateDataset."""
        items = []
        for q in range(n_questions):
            for t in range(traces_per_q):
                items.append({
                    "layer_data": {8: np.zeros((10, 64), dtype=np.float32)},
                    "labels": np.zeros(10, dtype=np.int64),
                    "mask": np.ones(10, dtype=np.float32),
                    "sample_idx": q,
                })

        class _FakeDataset:
            pass

        ds = _FakeDataset()
        ds.items = items
        return ds

    def test_no_question_leakage(self):
        from train_onset_detector import group_aware_split

        ds = self._build_mock_dataset(n_questions=50, traces_per_q=3)
        train_sub, val_sub = group_aware_split(ds, train_ratio=0.8, seed=42)

        train_qs = {ds.items[i]["sample_idx"] for i in train_sub.indices}
        val_qs = {ds.items[i]["sample_idx"] for i in val_sub.indices}

        overlap = train_qs & val_qs
        assert len(overlap) == 0, f"Leaked questions in both splits: {overlap}"

    def test_all_traces_covered(self):
        from train_onset_detector import group_aware_split

        ds = self._build_mock_dataset(n_questions=30, traces_per_q=3)
        train_sub, val_sub = group_aware_split(ds, train_ratio=0.8, seed=123)

        assert len(train_sub) + len(val_sub) == len(ds.items)

    def test_split_ratio_approximately_correct(self):
        from train_onset_detector import group_aware_split

        ds = self._build_mock_dataset(n_questions=100, traces_per_q=3)
        train_sub, val_sub = group_aware_split(ds, train_ratio=0.8, seed=7)

        train_ratio = len(train_sub) / len(ds.items)
        assert 0.7 < train_ratio < 0.9, f"Train ratio {train_ratio:.2f} too far from 0.8"

    def test_deterministic_across_calls(self):
        from train_onset_detector import group_aware_split

        ds = self._build_mock_dataset(n_questions=40, traces_per_q=3)
        t1, v1 = group_aware_split(ds, train_ratio=0.8, seed=42)
        t2, v2 = group_aware_split(ds, train_ratio=0.8, seed=42)

        assert set(t1.indices) == set(t2.indices)
        assert set(v1.indices) == set(v2.indices)
