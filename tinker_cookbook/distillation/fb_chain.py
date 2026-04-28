"""
Latent-state forward-backward (FB) messages for on-policy distillation.

We treat the student's sampled trajectory as an HMM over a small latent "track
state" ``z_t`` (by default binary: ``on-track`` / ``off-track``) with:

* **Emission**: local teacher-student agreement on the sampled token. When the
  student and teacher agree, the ``on-track`` state has high emission; the
  ``off-track`` state has a fixed background level, so disagreement naturally
  pushes posterior mass toward ``off-track``.
* **Transition**: structured by *waypoint boundaries* on the decoded text. At
  non-boundary steps, transitions are near-identity (state persists); at
  boundary steps, transitions interpolate toward uniform, i.e. the latent
  state may reset. This injects process structure (math steps, code blocks,
  agentic sub-goals) as a **prior** on where credit can change.
* **Terminal factor**: trajectory return ``R = sigmoid(rho * total_reward)``
  enters as the terminal β initialization, putting mass on ``on-track``
  proportional to ``R``.

Running the standard HMM forward-backward yields the posterior edge marginal
``gamma_t(z)``; we use ``gamma_t(on-track)`` (optionally powered by a trajectory
scalar ``R``) as the per-token KL weight in the student's distillation loss.

This is the computational core of an **outcome-conditioned posterior
regularization** view of on-policy distillation. Unlike a degenerate,
single-state FB (which is algebraically equivalent to cumulative emission up to
a constant), the latent chain lets β carry information that α cannot, and lets
the terminal reward modulate per-token weights beyond a sequence-constant
offset.

Backward-compatible primitives (``forward_log_messages``,
``backward_log_messages``, ``kl_weights_from_fb_tension``) are retained so the
degenerate single-state recipe remains available as an ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# --------------------------------------------------------------------------- #
# Low-level helpers                                                           #
# --------------------------------------------------------------------------- #


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp(min=1e-12))


def local_emission_from_logprob_diff(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    delta_scale: float,
) -> torch.Tensor:
    """Scalar ``on-track`` emission in (0, 1] from per-position logprob gap.

    Both tensors must have the same length (one value per generated token).

    Args:
        student_logprobs: Per-token ``log p_student(a_t | prefix)`` for sampled actions.
        teacher_logprobs: Per-token ``log p_teacher(a_t | prefix)`` on the same tokens.
        delta_scale: Positive constant; larger → sharper agreement requirement.

    Returns:
        Tensor of shape ``(T,)`` with values in ``(0, 1]``.
    """
    diff = (student_logprobs - teacher_logprobs).abs()
    return torch.exp(-delta_scale * diff).clamp(1e-8, 1.0)


def terminal_potential_from_reward(
    total_reward: float,
    reward_scale: float,
) -> float:
    """Map scalar trajectory return to a terminal factor ``R`` in ``(0, 1]``.

    Large positive rewards → ``R → 1``; large negative → ``R → 0``.
    """
    if reward_scale <= 0:
        reward_scale = 1.0
    r = 1.0 / (1.0 + math.exp(-float(reward_scale) * float(total_reward)))
    return max(r, 1e-8)


# --------------------------------------------------------------------------- #
# Legacy single-state FB (kept for backward compatibility / ablation).        #
# --------------------------------------------------------------------------- #


def forward_log_messages(emission: torch.Tensor) -> torch.Tensor:
    """Chain forward (single-state, legacy): ``log α[t] = sum_{k<=t} log e_k``."""
    log_e = _safe_log(emission)
    return torch.cumsum(log_e, dim=0)


def backward_log_messages(emission: torch.Tensor, terminal_r: float) -> torch.Tensor:
    """Chain backward with terminal ``R`` at the end of the sequence (legacy).

    ``log β[T-1] = log R`` and ``log β[t] = log β[t+1] + log e[t+1]``.
    """
    t_len = emission.shape[0]
    if t_len == 0:
        return emission
    log_e = _safe_log(emission)
    log_beta = emission.new_zeros(t_len)
    log_r = math.log(max(terminal_r, 1e-12))
    log_beta[t_len - 1] = log_r
    for t in range(t_len - 2, -1, -1):
        log_beta[t] = log_beta[t + 1] + log_e[t + 1]
    return log_beta


def kl_weights_from_fb_tension(
    log_alpha: torch.Tensor,
    log_beta: torch.Tensor,
    gate_temperature: float,
) -> torch.Tensor:
    """Legacy z-scored α/β gate. Note: in the single-state chain this reduces to
    a sigmoid of normalized cumulative emission, because ``log α + log β`` is
    constant in ``t``. Kept for ablation; prefer :func:`compute_latent_fb`.
    """
    if log_alpha.numel() == 0:
        return log_alpha
    z_a = (log_alpha - log_alpha.mean()) / (log_alpha.std(unbiased=False).clamp(min=1e-6))
    z_b = (log_beta - log_beta.mean()) / (log_beta.std(unbiased=False).clamp(min=1e-6))
    x = gate_temperature * (z_b - z_a)
    return torch.sigmoid(x)


# --------------------------------------------------------------------------- #
# Latent-state FB (core method).                                              #
# --------------------------------------------------------------------------- #


def build_latent_emission(
    agreement: torch.Tensor,
    emission_off_level: float,
    num_states: int = 2,
) -> torch.Tensor:
    """Per-token log-emission over ``K`` latent states.

    * State ``K-1`` (``on-track``) uses the teacher-student ``agreement`` directly.
    * State ``0`` (``off-track``) uses ``emission_off_level`` (constant).
    * For ``K > 2``, intermediate states linearly interpolate between the two,
      allowing finer-grained state ablations.

    Args:
        agreement: Shape ``(T,)`` in ``(0, 1]``, typically from
            :func:`local_emission_from_logprob_diff`.
        emission_off_level: Constant emission for the off-track state. Values in
            ``(0, 1)`` are recommended; smaller → disagreement more quickly
            pushes posterior toward off-track.
        num_states: Number of latent states ``K``. ``K = 2`` is the default
            paper configuration.

    Returns:
        Log-emission tensor of shape ``(T, K)``.
    """
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    t = agreement.shape[0]
    if t == 0:
        return agreement.new_zeros(0, num_states)
    agreement_c = agreement.clamp(1e-8, 1.0)
    if num_states == 1:
        return _safe_log(agreement_c).unsqueeze(-1)
    off = agreement.new_full((t,), float(emission_off_level))
    cols = []
    for k in range(num_states):
        w = k / (num_states - 1)
        mix = (1.0 - w) * off + w * agreement_c
        cols.append(_safe_log(mix))
    return torch.stack(cols, dim=-1)


def build_boundary_transition(
    seq_len: int,
    boundary_mask: torch.Tensor | None,
    flip_prob: float,
    boundary_reset_prob: float,
    num_states: int = 2,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-step log-transition induced by waypoint boundaries.

    At non-boundary steps, the transition matrix is ``stay`` on the diagonal and
    ``move`` uniformly elsewhere, with ``stay = 1 - flip_prob * (K-1)/K``. At
    boundary steps, the transition is mixed toward the uniform matrix with
    weight ``boundary_reset_prob``, allowing the latent state to reset at
    logical breakpoints.

    Args:
        seq_len: ``T``.
        boundary_mask: Shape ``(T,)`` float in ``{0, 1}``. ``boundary_mask[t] = 1``
            marks the end of a waypoint segment: the transition ``t -> t+1``
            uses the reset distribution. ``None`` → no boundaries.
        flip_prob: Per-step flip probability at non-boundary steps.
        boundary_reset_prob: Interpolation weight toward uniform at boundaries.
        num_states: ``K``.

    Returns:
        Tensor of shape ``(T-1, K, K)`` with log-probabilities. Empty when
        ``T <= 1``.
    """
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    k = num_states
    dtype_ = dtype or torch.float32

    if seq_len <= 1:
        return torch.zeros((0, k, k), dtype=dtype_, device=device)

    flip = float(max(0.0, min(1.0, flip_prob)))
    reset = float(max(0.0, min(1.0, boundary_reset_prob)))

    if k == 1:
        trans_nb = torch.ones((1, 1), dtype=dtype_, device=device)
        trans_b = trans_nb.clone()
    else:
        stay = 1.0 - flip * (k - 1) / k
        move = flip / k
        trans_nb = torch.full((k, k), move, dtype=dtype_, device=device)
        trans_nb.fill_diagonal_(stay)
        uniform = torch.full((k, k), 1.0 / k, dtype=dtype_, device=device)
        trans_b = (1.0 - reset) * trans_nb + reset * uniform

    log_nb = _safe_log(trans_nb)
    log_b = _safe_log(trans_b)

    t_minus = seq_len - 1
    if boundary_mask is None:
        return log_nb.unsqueeze(0).expand(t_minus, k, k).contiguous()

    bm = boundary_mask.to(dtype=dtype_, device=device)
    if bm.shape[0] != seq_len:
        raise ValueError(
            f"boundary_mask length {bm.shape[0]} does not match seq_len {seq_len}"
        )
    gate = bm[:t_minus].view(t_minus, 1, 1)
    return (1.0 - gate) * log_nb + gate * log_b


def latent_forward_backward(
    log_emission: torch.Tensor,
    log_transition: torch.Tensor,
    log_prior: torch.Tensor,
    log_terminal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-space HMM forward-backward with a terminal log-potential folded
    into the last ``β`` step.

    Args:
        log_emission: ``(T, K)``.
        log_transition: ``(T-1, K, K)`` (time-varying) or ``(K, K)`` (stationary).
        log_prior: ``(K,)`` initial state log-probabilities.
        log_terminal: ``(K,)`` terminal log-potential; sets ``β[T-1] = log_terminal``.

    Returns:
        Tuple ``(log_alpha, log_beta, log_gamma)`` each of shape ``(T, K)``,
        where ``log_gamma`` is the row-normalized log posterior edge marginal.
    """
    t_len, k = log_emission.shape
    if t_len == 0:
        z = log_emission.new_zeros(0, k)
        return z, z, z

    stationary = log_transition.dim() == 2

    log_alpha = log_emission.new_zeros(t_len, k)
    log_alpha[0] = log_prior + log_emission[0]
    for t in range(1, t_len):
        trans_t = log_transition if stationary else log_transition[t - 1]
        log_alpha[t] = (
            torch.logsumexp(log_alpha[t - 1].unsqueeze(-1) + trans_t, dim=0)
            + log_emission[t]
        )

    log_beta = log_emission.new_zeros(t_len, k)
    log_beta[t_len - 1] = log_terminal
    for t in range(t_len - 2, -1, -1):
        trans_t = log_transition if stationary else log_transition[t]
        log_beta[t] = torch.logsumexp(
            trans_t + (log_emission[t + 1] + log_beta[t + 1]).unsqueeze(0),
            dim=-1,
        )

    log_ab = log_alpha + log_beta
    log_gamma = log_ab - torch.logsumexp(log_ab, dim=-1, keepdim=True)
    return log_alpha, log_beta, log_gamma


# --------------------------------------------------------------------------- #
# High-level API.                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class LatentFBResult:
    """Richer output of :func:`compute_latent_fb`."""

    kl_weights: torch.Tensor  #: ``(T,)`` per-token KL multiplier in ``[0, 1]``.
    log_alpha: torch.Tensor  #: ``(T, K)``
    log_beta: torch.Tensor  #: ``(T, K)``
    log_gamma: torch.Tensor  #: ``(T, K)`` log posterior edge marginal.
    emission: torch.Tensor  #: ``(T, K)`` emission probabilities (not log).
    terminal_reward: float  #: ``R = sigmoid(rho * total_reward)``.


def _build_priorlike(
    on_track_mass: float,
    num_states: int,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Place ``on_track_mass`` on state ``K-1``, split the rest uniformly across
    the remaining states, and return the normalized vector (shape ``(K,)``).
    """
    on = float(max(0.0, min(1.0, on_track_mass)))
    if num_states == 1:
        vec = torch.ones(1, device=device, dtype=dtype)
    else:
        off_share = (1.0 - on) / (num_states - 1)
        vec = torch.full((num_states,), off_share, device=device, dtype=dtype)
        vec[num_states - 1] = on
    return vec / vec.sum()


def compute_latent_fb(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    total_reward: float,
    *,
    emission_delta_scale: float,
    terminal_reward_scale: float,
    boundary_mask: torch.Tensor | None = None,
    num_states: int = 2,
    emission_off_level: float = 0.3,
    transition_flip_prob: float = 0.05,
    boundary_reset_prob: float = 0.5,
    gate_exponent_gamma: float = 1.0,
    gate_exponent_reward: float = 0.0,
    prior_on_track: float = 0.5,
) -> LatentFBResult:
    """Full latent-state FB + posterior-regularized KL weights.

    The per-token weight is::

        w_t = gamma_t(on-track) ** gate_exponent_gamma
              * R ** gate_exponent_reward

    ``R`` already enters through the terminal factor, so
    ``gate_exponent_reward = 0`` is a clean "only-through-posterior" setting;
    set ``> 0`` for stronger outcome gating.

    Args:
        student_logprobs: Shape ``(T,)``.
        teacher_logprobs: Shape ``(T,)``, aligned with the sampled tokens.
        total_reward: Scalar trajectory return.
        emission_delta_scale: Passed to :func:`local_emission_from_logprob_diff`.
        terminal_reward_scale: Passed to :func:`terminal_potential_from_reward`.
        boundary_mask: ``(T,)`` float in ``{0, 1}``; waypoint boundaries inducing
            latent-state resets via the transition matrix. ``None`` → stationary.
        num_states: Latent state count ``K``. ``2`` is the paper default.
        emission_off_level: Constant emission for the off-track state.
        transition_flip_prob: Non-boundary per-step flip probability.
        boundary_reset_prob: Boundary-step mix weight toward uniform.
        gate_exponent_gamma: Exponent on ``gamma_t(on-track)`` in the weight.
        gate_exponent_reward: Exponent on the multiplicative ``R`` injection.
        prior_on_track: Initial probability of being on-track.

    Returns:
        :class:`LatentFBResult`.
    """
    t_len = student_logprobs.shape[0]
    k = num_states
    device = student_logprobs.device
    dtype = student_logprobs.dtype

    if t_len == 0:
        z = student_logprobs.new_zeros(0, k)
        return LatentFBResult(
            kl_weights=student_logprobs.new_zeros(0),
            log_alpha=z,
            log_beta=z,
            log_gamma=z,
            emission=z,
            terminal_reward=1.0,
        )

    agreement = local_emission_from_logprob_diff(
        student_logprobs, teacher_logprobs, emission_delta_scale
    )
    log_emission = build_latent_emission(agreement, emission_off_level, k)
    emission = torch.exp(log_emission)

    log_trans = build_boundary_transition(
        seq_len=t_len,
        boundary_mask=boundary_mask,
        flip_prob=transition_flip_prob,
        boundary_reset_prob=boundary_reset_prob,
        num_states=k,
        device=device,
        dtype=dtype,
    )

    prior = _build_priorlike(prior_on_track, k, device=device, dtype=dtype)
    log_prior = _safe_log(prior)

    r = terminal_potential_from_reward(total_reward, terminal_reward_scale)
    terminal = _build_priorlike(r, k, device=device, dtype=dtype)
    log_terminal = _safe_log(terminal)

    log_alpha, log_beta, log_gamma = latent_forward_backward(
        log_emission, log_trans, log_prior, log_terminal
    )

    log_gamma_on = log_gamma[:, k - 1]
    weights = torch.exp(gate_exponent_gamma * log_gamma_on)
    if gate_exponent_reward != 0.0:
        weights = weights * (r ** float(gate_exponent_reward))
    weights = weights.clamp(0.0, 1.0)

    return LatentFBResult(
        kl_weights=weights,
        log_alpha=log_alpha,
        log_beta=log_beta,
        log_gamma=log_gamma,
        emission=emission,
        terminal_reward=r,
    )


def compute_fb_kl_weights_and_messages(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    total_reward: float,
    *,
    emission_delta_scale: float,
    terminal_reward_scale: float,
    gate_temperature: float,
    boundary_mask: torch.Tensor | None = None,
    num_states: int = 2,
    emission_off_level: float = 0.3,
    transition_flip_prob: float = 0.05,
    boundary_reset_prob: float = 0.5,
    gate_exponent_gamma: float = 1.0,
    gate_exponent_reward: float = 0.0,
    prior_on_track: float = 0.5,
    use_latent_fb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Back-compatible entry point returning ``(T,)``-shaped diagnostics.

    When ``use_latent_fb`` is ``True`` (default), runs the latent-state FB with
    optional waypoint transitions and multiplicative terminal reward. The
    returned ``log_alpha`` / ``log_beta`` / ``emission`` are the on-track
    slices of the full ``(T, K)`` tensors, for drop-in compatibility with
    existing call sites that average per-token.

    When ``use_latent_fb`` is ``False``, falls back to the legacy single-state
    FB plus z-scored α/β gate (kept for ablation; ``gate_temperature`` is used
    only in this branch).
    """
    if student_logprobs.shape[0] == 0:
        z = student_logprobs
        return z, z, z, z

    if not use_latent_fb:
        emission = local_emission_from_logprob_diff(
            student_logprobs, teacher_logprobs, emission_delta_scale
        )
        r = terminal_potential_from_reward(total_reward, terminal_reward_scale)
        log_alpha = forward_log_messages(emission)
        log_beta = backward_log_messages(emission, r)
        kl_w = kl_weights_from_fb_tension(log_alpha, log_beta, gate_temperature)
        return kl_w, log_alpha, log_beta, emission

    result = compute_latent_fb(
        student_logprobs,
        teacher_logprobs,
        total_reward,
        emission_delta_scale=emission_delta_scale,
        terminal_reward_scale=terminal_reward_scale,
        boundary_mask=boundary_mask,
        num_states=num_states,
        emission_off_level=emission_off_level,
        transition_flip_prob=transition_flip_prob,
        boundary_reset_prob=boundary_reset_prob,
        gate_exponent_gamma=gate_exponent_gamma,
        gate_exponent_reward=gate_exponent_reward,
        prior_on_track=prior_on_track,
    )
    on = num_states - 1
    return (
        result.kl_weights,
        result.log_alpha[:, on],
        result.log_beta[:, on],
        result.emission[:, on],
    )
