"""Tests for WGHD training engine components."""

import torch

from tinker_cookbook.distillation.train_wghd import (
    Config,
    WaypointConfig,
    _estimate_token_boundaries,
)
from tinker_cookbook.distillation.waypoint import (
    WaypointStore,
    detect_boundary_positions,
)


class TestEstimateTokenBoundaries:
    def test_basic_mapping(self):
        text = "Hello world, this is a test sentence."
        char_positions = [6, 13, 21]
        boundaries = _estimate_token_boundaries(text, char_positions, total_tokens=10)
        assert len(boundaries) == 3
        for b in boundaries:
            assert 0 <= b < 10

    def test_empty_text(self):
        boundaries = _estimate_token_boundaries("", [], total_tokens=0)
        assert boundaries == []

    def test_proportional_mapping(self):
        text = "A" * 100
        char_positions = [25, 50, 75]
        boundaries = _estimate_token_boundaries(text, char_positions, total_tokens=100)
        assert boundaries[0] == 25
        assert boundaries[1] == 50
        assert boundaries[2] == 75

    def test_boundary_clamping(self):
        text = "Short"
        char_positions = [0, 100]
        boundaries = _estimate_token_boundaries(text, char_positions, total_tokens=5)
        for b in boundaries:
            assert 0 <= b < 5


class TestWaypointConfig:
    def test_default_config(self):
        config = WaypointConfig()
        assert config.enabled is True
        assert config.task_type == "math"
        assert config.similarity_threshold == 0.5
        assert config.gate_temperature == 5.0
        assert config.step_reward == 0.1

    def test_custom_config(self):
        config = WaypointConfig(
            enabled=True,
            task_type="code",
            similarity_threshold=0.3,
            gate_temperature=10.0,
            step_reward=0.2,
        )
        assert config.task_type == "code"
        assert config.similarity_threshold == 0.3


class TestWaypointIntegration:
    """Integration tests for the waypoint + boundary detection pipeline."""

    def test_math_text_boundary_detection(self):
        math_text = (
            "Let's solve this step by step.\n\n"
            "Step 1: We know that x + 5 = 10\n"
            "Step 2: Subtracting 5 from both sides: x = 5\n\n"
            "Therefore, x = 5.\n"
            "The answer is \\boxed{5}"
        )
        positions = detect_boundary_positions(math_text, task_type="math", min_gap_chars=0)
        assert len(positions) >= 3

    def test_code_text_boundary_detection(self):
        code_text = (
            "```python\n"
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    result = 0\n"
            "    for i in range(2, n + 1):\n"
            "        result = result + i\n"
            "    return result\n"
            "```"
        )
        positions = detect_boundary_positions(code_text, task_type="code", min_gap_chars=0)
        assert len(positions) >= 3

    def test_empty_waypoint_store_graceful(self):
        store = WaypointStore()
        assert store.get("any prompt") is None
        assert len(store) == 0
