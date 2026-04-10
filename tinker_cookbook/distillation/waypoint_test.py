"""Tests for waypoint extraction and similarity computation."""

import json
import tempfile
from pathlib import Path

import torch

from tinker_cookbook.distillation.waypoint import (
    WaypointFeature,
    WaypointSequence,
    WaypointStore,
    compute_adaptive_kl_weight,
    compute_segment_kl_weights,
    compute_waypoint_similarity,
    detect_boundary_positions,
    find_active_waypoint_index,
)


class TestDetectBoundaryPositions:
    def test_math_double_newline(self):
        text = "First line\n\nSecond paragraph\n\nThird paragraph"
        positions = detect_boundary_positions(text, task_type="math")
        assert len(positions) >= 1
        assert 10 in positions or 11 in positions  # around the first \n\n

    def test_math_step_markers(self):
        text = "Step 1: Calculate x\nStep 2: Substitute\nStep 3: Solve"
        positions = detect_boundary_positions(text, task_type="math", min_gap_chars=0)
        assert len(positions) >= 2

    def test_math_conclusion_markers(self):
        text = "We have x = 5. " + "a" * 60 + "Therefore, the answer is 10."
        positions = detect_boundary_positions(text, task_type="math", min_gap_chars=0)
        assert len(positions) >= 1

    def test_code_boundaries(self):
        text = "def foo():\n    for i in range(10):\n        if i > 5:\n            return i"
        positions = detect_boundary_positions(text, task_type="code", min_gap_chars=0)
        assert len(positions) >= 3

    def test_min_gap_filtering(self):
        text = "Step 1: A\nStep 2: B\nStep 3: C"
        positions_no_gap = detect_boundary_positions(text, task_type="math", min_gap_chars=0)
        positions_with_gap = detect_boundary_positions(text, task_type="math", min_gap_chars=100)
        assert len(positions_with_gap) <= len(positions_no_gap)

    def test_empty_text(self):
        positions = detect_boundary_positions("", task_type="math")
        assert positions == []

    def test_no_boundaries(self):
        positions = detect_boundary_positions("just a simple sentence", task_type="math")
        assert positions == []


class TestWaypointFeature:
    def test_to_dense_vector(self):
        wp = WaypointFeature(
            token_index=10,
            top_k_indices=[100, 200, 300],
            top_k_logprobs=[-1.0, -2.0, -3.0],
        )
        vec = wp.to_dense_vector()
        assert vec.shape == (3,)
        assert torch.allclose(vec.sum(), torch.tensor(1.0), atol=1e-5)
        assert vec[0] > vec[1] > vec[2]

    def test_serialization_roundtrip(self):
        wp = WaypointFeature(
            token_index=42,
            top_k_indices=[1, 2, 3, 4, 5],
            top_k_logprobs=[-0.5, -1.0, -1.5, -2.0, -3.0],
            char_offset=100,
        )
        d = wp.to_dict()
        wp2 = WaypointFeature.from_dict(d)
        assert wp.token_index == wp2.token_index
        assert wp.top_k_indices == wp2.top_k_indices
        assert wp.top_k_logprobs == wp2.top_k_logprobs
        assert wp.char_offset == wp2.char_offset


class TestWaypointSequence:
    def test_serialization_roundtrip(self):
        seq = WaypointSequence(
            prompt="What is 2+2?",
            waypoints=[
                WaypointFeature(10, [1, 2], [-1.0, -2.0]),
                WaypointFeature(20, [3, 4], [-0.5, -1.5]),
            ],
            task_type="math",
            full_response="The answer is 4",
        )
        d = seq.to_dict()
        seq2 = WaypointSequence.from_dict(d)
        assert seq.prompt == seq2.prompt
        assert len(seq.waypoints) == len(seq2.waypoints)
        assert seq.task_type == seq2.task_type
        assert seq.full_response == seq2.full_response


class TestWaypointStore:
    def test_add_and_get(self):
        store = WaypointStore()
        seq = WaypointSequence(
            prompt="test prompt",
            waypoints=[WaypointFeature(5, [10], [-1.0])],
        )
        store.add(seq)
        assert len(store) == 1
        assert "test prompt" in store
        retrieved = store.get("test prompt")
        assert retrieved is not None
        assert retrieved.prompt == "test prompt"

    def test_missing_key(self):
        store = WaypointStore()
        assert store.get("nonexistent") is None
        assert "nonexistent" not in store

    def test_save_and_load(self):
        store = WaypointStore()
        store.add(WaypointSequence(
            prompt="q1",
            waypoints=[WaypointFeature(5, [10, 20], [-1.0, -2.0])],
            task_type="math",
        ))
        store.add(WaypointSequence(
            prompt="q2",
            waypoints=[
                WaypointFeature(3, [30], [-0.5]),
                WaypointFeature(8, [40], [-1.5]),
            ],
            task_type="code",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "waypoints.jsonl"
            store.save(path)
            loaded = WaypointStore.load(path)

        assert len(loaded) == 2
        q1 = loaded.get("q1")
        assert q1 is not None
        assert len(q1.waypoints) == 1
        q2 = loaded.get("q2")
        assert q2 is not None
        assert len(q2.waypoints) == 2
        assert q2.task_type == "code"


class TestComputeWaypointSimilarity:
    def test_identical_distributions(self):
        wp = WaypointFeature(
            token_index=10,
            top_k_indices=[0, 1, 2],
            top_k_logprobs=[-1.0, -2.0, -3.0],
        )
        student_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        sim = compute_waypoint_similarity(student_logprobs, wp)
        assert sim > 0.99

    def test_different_distributions(self):
        wp = WaypointFeature(
            token_index=10,
            top_k_indices=[0, 1, 2],
            top_k_logprobs=[-0.1, -10.0, -10.0],
        )
        student_logprobs = torch.tensor([-10.0, -10.0, -0.1])
        sim = compute_waypoint_similarity(student_logprobs, wp)
        assert sim < 0.5

    def test_with_full_vocab_logprobs(self):
        wp = WaypointFeature(
            token_index=10,
            top_k_indices=[5, 10, 15],
            top_k_logprobs=[-1.0, -2.0, -3.0],
        )
        student_logprobs = torch.randn(100)
        student_logprobs[5] = -1.0
        student_logprobs[10] = -2.0
        student_logprobs[15] = -3.0
        sim = compute_waypoint_similarity(student_logprobs, wp)
        assert -1.0 <= sim <= 1.0


class TestComputeAdaptiveKlWeight:
    def test_high_similarity_low_weight(self):
        weight = compute_adaptive_kl_weight(similarity=0.9, threshold=0.5, temperature=5.0)
        assert weight < 0.2

    def test_low_similarity_high_weight(self):
        weight = compute_adaptive_kl_weight(similarity=0.1, threshold=0.5, temperature=5.0)
        assert weight > 0.8

    def test_at_threshold(self):
        weight = compute_adaptive_kl_weight(similarity=0.5, threshold=0.5, temperature=5.0)
        assert abs(weight - 0.5) < 0.01

    def test_temperature_sharpness(self):
        w_sharp = compute_adaptive_kl_weight(0.3, 0.5, temperature=20.0)
        w_smooth = compute_adaptive_kl_weight(0.3, 0.5, temperature=1.0)
        assert w_sharp > w_smooth

    def test_output_range(self):
        for sim in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]:
            weight = compute_adaptive_kl_weight(sim, threshold=0.5)
            assert 0.0 <= weight <= 1.0


class TestFindActiveWaypointIndex:
    def test_before_first(self):
        waypoints = [
            WaypointFeature(10, [], []),
            WaypointFeature(20, [], []),
        ]
        assert find_active_waypoint_index(5, waypoints) == 0

    def test_between_waypoints(self):
        waypoints = [
            WaypointFeature(10, [], []),
            WaypointFeature(20, [], []),
            WaypointFeature(30, [], []),
        ]
        assert find_active_waypoint_index(15, waypoints) == 1

    def test_after_all(self):
        waypoints = [
            WaypointFeature(10, [], []),
            WaypointFeature(20, [], []),
        ]
        assert find_active_waypoint_index(25, waypoints) == 1

    def test_at_waypoint(self):
        waypoints = [
            WaypointFeature(10, [], []),
            WaypointFeature(20, [], []),
        ]
        assert find_active_waypoint_index(10, waypoints) == 1


class TestComputeSegmentKlWeights:
    def test_no_waypoints(self):
        weights, reward = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=[],
            segment_similarities=[],
        )
        assert weights.shape == (100,)
        assert torch.all(weights == 1.0)
        assert reward == 0.0

    def test_all_matched(self):
        waypoints = [
            WaypointFeature(25, [], []),
            WaypointFeature(50, [], []),
            WaypointFeature(75, [], []),
        ]
        sims = [0.9, 0.8, 0.7]
        weights, reward = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=waypoints,
            segment_similarities=sims,
            threshold=0.5,
            step_reward=0.1,
        )
        assert weights.shape == (100,)
        assert abs(reward - 0.3) < 1e-9

    def test_none_matched(self):
        waypoints = [
            WaypointFeature(25, [], []),
            WaypointFeature(50, [], []),
        ]
        sims = [0.1, 0.2]
        weights, reward = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=waypoints,
            segment_similarities=sims,
            threshold=0.5,
            step_reward=0.1,
        )
        assert reward == 0.0

    def test_mixed_matching(self):
        waypoints = [
            WaypointFeature(30, [], []),
            WaypointFeature(60, [], []),
        ]
        sims = [0.8, 0.2]
        weights, reward = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=waypoints,
            segment_similarities=sims,
            threshold=0.5,
            step_reward=0.1,
        )
        assert reward == 0.1

    def test_kl_weights_vary_by_segment(self):
        waypoints = [
            WaypointFeature(50, [], []),
        ]
        sims_high = [0.9]
        sims_low = [0.1]
        weights_high, _ = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=waypoints,
            segment_similarities=sims_high,
            threshold=0.5,
        )
        weights_low, _ = compute_segment_kl_weights(
            student_token_count=100,
            waypoints=waypoints,
            segment_similarities=sims_low,
            threshold=0.5,
        )
        assert weights_high.mean() < weights_low.mean()
