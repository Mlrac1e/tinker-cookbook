"""Unit tests for token-chain forward-backward helpers."""

import math

import torch

from tinker_cookbook.distillation.fb_chain import (
    LatentFBResult,
    backward_log_messages,
    build_boundary_transition,
    build_latent_emission,
    compute_fb_kl_weights_and_messages,
    compute_latent_fb,
    forward_log_messages,
    latent_forward_backward,
    local_emission_from_logprob_diff,
    terminal_potential_from_reward,
)

# --------------------------------------------------------------------------- #
# Legacy single-state primitives (kept for back-compat / ablation).           #
# --------------------------------------------------------------------------- #


class TestLegacyPrimitives:
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
        assert torch.allclose(log_b[2], torch.tensor(math.log(0.5)))

    def test_emission_agreement(self):
        x = torch.zeros(4)
        y = torch.zeros(4)
        e = local_emission_from_logprob_diff(x, y, delta_scale=1.0)
        assert torch.allclose(e, torch.ones_like(e))

    def test_terminal_sigmoid(self):
        r = terminal_potential_from_reward(5.0, reward_scale=1.0)
        assert 0 < r <= 1

    def test_legacy_branch_preserved(self):
        s = torch.tensor([-1.0, -2.0, -0.5])
        t = torch.tensor([-1.1, -1.9, -0.4])
        w, la, lb, em = compute_fb_kl_weights_and_messages(
            s,
            t,
            total_reward=1.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
            gate_temperature=2.0,
            use_latent_fb=False,
        )
        assert w.shape == s.shape and em.shape == s.shape
        assert la.shape == s.shape and lb.shape == s.shape
        assert (w >= 0).all() and (w <= 1).all()


# --------------------------------------------------------------------------- #
# Latent-state FB primitives.                                                 #
# --------------------------------------------------------------------------- #


class TestLatentPrimitives:
    def test_build_latent_emission_shape_and_order(self):
        agreement = torch.tensor([1.0, 0.5, 0.1])
        log_e = build_latent_emission(agreement, emission_off_level=0.3, num_states=2)
        assert log_e.shape == (3, 2)
        # On-track (col 1) should equal log(agreement); off-track constant.
        assert torch.allclose(log_e[:, 1], torch.log(agreement.clamp(min=1e-12)))
        assert torch.allclose(log_e[:, 0], torch.full((3,), math.log(0.3)), atol=1e-6)

    def test_build_latent_emission_single_state(self):
        agreement = torch.tensor([1.0, 0.5])
        log_e = build_latent_emission(agreement, emission_off_level=0.2, num_states=1)
        assert log_e.shape == (2, 1)

    def test_boundary_transition_shape(self):
        log_t = build_boundary_transition(
            seq_len=5,
            boundary_mask=None,
            flip_prob=0.05,
            boundary_reset_prob=0.5,
            num_states=2,
        )
        assert log_t.shape == (4, 2, 2)
        row_sums = torch.logsumexp(log_t, dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5)

    def test_boundary_transition_resets_at_boundary(self):
        bm = torch.tensor([0.0, 1.0, 0.0, 0.0])
        log_t = build_boundary_transition(
            seq_len=4,
            boundary_mask=bm,
            flip_prob=0.01,
            boundary_reset_prob=1.0,
            num_states=2,
        )
        # Transition from t=1 (boundary) should be uniform (log 0.5).
        assert torch.allclose(log_t[1], torch.full((2, 2), math.log(0.5)), atol=1e-6)
        # Non-boundary transition should be near-identity on the diagonal.
        assert log_t[0, 0, 0] > log_t[0, 0, 1]
        assert log_t[0, 1, 1] > log_t[0, 1, 0]

    def test_boundary_transition_zero_len(self):
        log_t = build_boundary_transition(
            seq_len=1, boundary_mask=None, flip_prob=0.1, boundary_reset_prob=0.5,
        )
        assert log_t.shape == (0, 2, 2)

    def test_latent_fb_rows_sum_to_one(self):
        agreement = torch.tensor([0.9, 0.4, 0.2, 0.7])
        log_e = build_latent_emission(agreement, emission_off_level=0.3, num_states=2)
        log_t = build_boundary_transition(
            seq_len=4, boundary_mask=None, flip_prob=0.05, boundary_reset_prob=0.5
        )
        prior = torch.log(torch.tensor([0.5, 0.5]))
        term = torch.log(torch.tensor([0.3, 0.7]))
        la, lb, lg = latent_forward_backward(log_e, log_t, prior, term)
        assert la.shape == lb.shape == lg.shape == (4, 2)
        row_sums = torch.logsumexp(lg, dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5)


# --------------------------------------------------------------------------- #
# Latent FB high-level API behavior.                                          #
# --------------------------------------------------------------------------- #


class TestLatentFBHighLevel:
    def _default_call(self, reward: float = 1.0, **overrides) -> LatentFBResult:
        s = torch.tensor([-1.0, -2.0, -0.5, -1.5])
        t = torch.tensor([-1.1, -1.9, -0.4, -1.6])
        kw = {
            "emission_delta_scale": 1.0,
            "terminal_reward_scale": 1.0,
        }
        kw.update(overrides)
        return compute_latent_fb(s, t, reward, **kw)

    def test_shapes_and_weight_range(self):
        result = self._default_call()
        t_len = 4
        assert result.kl_weights.shape == (t_len,)
        assert result.log_alpha.shape == (t_len, 2)
        assert result.log_beta.shape == (t_len, 2)
        assert result.log_gamma.shape == (t_len, 2)
        assert result.emission.shape == (t_len, 2)
        assert (result.kl_weights >= 0).all() and (result.kl_weights <= 1).all()
        assert 0 < result.terminal_reward <= 1

    def test_terminal_reward_actually_affects_weights(self):
        # Key invariant broken in the legacy z-scored version: R must change
        # per-token weights. Here it flows through the terminal β factor, so
        # even without multiplicative injection the weights must differ.
        r_hi = self._default_call(reward=5.0)
        r_lo = self._default_call(reward=-5.0)
        assert not torch.allclose(r_hi.kl_weights, r_lo.kl_weights, atol=1e-4)
        # High reward should push weights up (more on-track posterior mass).
        assert r_hi.kl_weights.mean() > r_lo.kl_weights.mean()

    def test_multiplicative_reward_injection(self):
        base = self._default_call(reward=0.0, gate_exponent_reward=0.0)
        amped = self._default_call(reward=5.0, gate_exponent_reward=2.0)
        # Amped should have both higher R and a larger multiplier.
        assert amped.terminal_reward > base.terminal_reward
        assert amped.kl_weights.mean() > 0  # sanity
        # With same reward, exponent must scale weights monotonically.
        w_a = self._default_call(reward=1.0, gate_exponent_reward=0.0).kl_weights
        w_b = self._default_call(reward=1.0, gate_exponent_reward=1.0).kl_weights
        # γ^1 * R^0 vs γ^1 * R^1 with R<1 → w_b <= w_a pointwise.
        assert torch.all(w_b <= w_a + 1e-6)

    def test_perfect_agreement_yields_on_track(self):
        # When student == teacher, emission = 1 on on-track, 0.3 on off-track
        # → posterior strongly favors on-track → γ_on ≈ 1.
        s = torch.zeros(5)
        t = torch.zeros(5)
        result = compute_latent_fb(
            s,
            t,
            total_reward=5.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
        )
        gamma_on = torch.exp(result.log_gamma[:, 1])
        assert (gamma_on > 0.9).all()

    def test_disagreement_shifts_posterior_off_track(self):
        # Large disagreement should depress γ_on.
        s = torch.tensor([-5.0, -5.0, -5.0, -5.0])
        t = torch.tensor([-0.5, -0.5, -0.5, -0.5])
        r_agree = compute_latent_fb(
            torch.zeros(4), torch.zeros(4),
            total_reward=0.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
        )
        r_disagree = compute_latent_fb(
            s, t,
            total_reward=0.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
        )
        gamma_on_a = torch.exp(r_agree.log_gamma[:, 1]).mean()
        gamma_on_d = torch.exp(r_disagree.log_gamma[:, 1]).mean()
        assert gamma_on_d < gamma_on_a - 0.1

    def test_boundary_mask_changes_posterior(self):
        s = torch.tensor([0.0, 0.0, -3.0, -3.0, 0.0, 0.0])
        t = torch.tensor([0.0, 0.0, -0.1, -0.1, 0.0, 0.0])
        # Without boundaries, state is sticky → the disagreement spreads.
        r_no = compute_latent_fb(
            s, t, total_reward=1.0,
            emission_delta_scale=1.0, terminal_reward_scale=1.0,
            boundary_mask=None,
        )
        # With a boundary right after the disagreement region, the posterior
        # should recover on-track mass more quickly in the tail.
        bm = torch.zeros(6)
        bm[3] = 1.0
        r_bnd = compute_latent_fb(
            s, t, total_reward=1.0,
            emission_delta_scale=1.0, terminal_reward_scale=1.0,
            boundary_mask=bm, boundary_reset_prob=1.0,
        )
        tail_no = torch.exp(r_no.log_gamma[4:, 1]).mean()
        tail_bnd = torch.exp(r_bnd.log_gamma[4:, 1]).mean()
        assert tail_bnd > tail_no

    def test_empty_sequence(self):
        s = torch.zeros(0)
        t = torch.zeros(0)
        result = compute_latent_fb(
            s, t, total_reward=1.0,
            emission_delta_scale=1.0, terminal_reward_scale=1.0,
        )
        assert result.kl_weights.numel() == 0
        assert result.log_alpha.shape == (0, 2)

    def test_num_states_ablation(self):
        # Going to 3 states should still produce valid posteriors.
        s = torch.tensor([-1.0, -2.0, -0.5, -1.5])
        t = torch.tensor([-1.1, -1.9, -0.4, -1.6])
        result = compute_latent_fb(
            s, t, total_reward=1.0,
            emission_delta_scale=1.0, terminal_reward_scale=1.0,
            num_states=3,
        )
        assert result.log_gamma.shape == (4, 3)
        row_sums = torch.logsumexp(result.log_gamma, dim=-1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5)


# --------------------------------------------------------------------------- #
# Back-compat wrapper (compute_fb_kl_weights_and_messages).                   #
# --------------------------------------------------------------------------- #


class TestBackCompatWrapper:
    def test_latent_default_shapes(self):
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
        assert w.shape == s.shape and em.shape == s.shape
        assert la.shape == s.shape and lb.shape == s.shape
        assert (w >= 0).all() and (w <= 1).all()

    def test_boundary_mask_accepted(self):
        s = torch.tensor([-1.0, -2.0, -0.5, -1.5])
        t = torch.tensor([-1.1, -1.9, -0.4, -1.6])
        bm = torch.tensor([0.0, 1.0, 0.0, 0.0])
        w, *_ = compute_fb_kl_weights_and_messages(
            s,
            t,
            total_reward=1.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
            gate_temperature=2.0,
            boundary_mask=bm,
        )
        assert w.shape == s.shape
        assert (w >= 0).all() and (w <= 1).all()

    def test_empty_wrapper(self):
        s = torch.zeros(0)
        t = torch.zeros(0)
        w, la, lb, em = compute_fb_kl_weights_and_messages(
            s,
            t,
            total_reward=1.0,
            emission_delta_scale=1.0,
            terminal_reward_scale=1.0,
            gate_temperature=2.0,
        )
        assert w.numel() == 0 and la.numel() == 0
        assert lb.numel() == 0 and em.numel() == 0
