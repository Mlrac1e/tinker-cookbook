"""Unit tests for token-chain forward–backward helpers."""

import torch

from tinker_cookbook.distillation.fb_chain import (
    backward_log_messages,
    compute_fb_kl_weights_and_messages,
    forward_log_messages,
    local_emission_from_logprob_diff,
    terminal_potential_from_reward,
)


class TestForwardBackwardChain:
    def test_forward_cumsum(self):
        e = torch.tensor([0.5, 0.5, 0.5])
        log_a = forward_log_messages(e)
        assert log_a.shape == e.shape
        assert torch.allclose(log_a[0], torch.log(e[0]))
        assert torch.allclose(log_a[2], torch.log(e).sum())

    def test_backward_terminal(self):
        e = torch.ones(3) * 0.8
        log_b = backward_log_messages(e, terminal_r=0.5)
        assert log_b.shape == e.shape
        assert torch.isfinite(log_b).all()

    def test_fb_weights_shape_and_range(self):
        s = torch.tensor([-1.0, -2.0, -0.5])
        t = torch.tensor([-1.1, -1.9, -0.4])
        w, la, lb, em = compute_fb_kl_weights_and_messages(
            s,
            t,
            total_reward=1.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
            gate_temperature=2.0,
        )
        assert w.shape == s.shape
        assert la.shape == s.shape
        assert lb.shape == s.shape
        assert em.shape == s.shape
        assert (w >= 0).all() and (w <= 1).all()

    def test_emission_agreement(self):
        x = torch.zeros(4)
        y = torch.zeros(4)
        e = local_emission_from_logprob_diff(x, y, delta_scale=1.0)
        assert torch.allclose(e, torch.ones_like(e))

    def test_terminal_sigmoid(self):
        r = terminal_potential_from_reward(5.0, reward_scale=1.0)
        assert 0 < r <= 1

