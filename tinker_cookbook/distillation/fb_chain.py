"""
Token-chain forward–backward messages for on-policy distillation.

We model the student's sampled trajectory as a length-T chain with one scalar
emission potential per position (local teacher–student agreement on the *sampled*
token). This yields explicit forward messages α and backward messages β in log
domain; β is initialized from a terminal potential derived from the trajectory
reward (outcome at the end of the episode).

This is the computational core for an FB-centric story: α encodes cumulative
local (prefix) consistency with the teacher; β encodes how much terminal mass
remains when conditioning on the future (including the final outcome).
"""

from __future__ import annotations

import math

import torch


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp(min=1e-12))


def local_emission_from_logprob_diff(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    delta_scale: float,
) -> torch.Tensor:
    """Scalar emission in (0, 1] from per-position logprob gap on the sampled token.

    Both tensors must have the same length (one value per generated token).

    Args:
        student_logprobs: Per-token log p_student(a_t | prefix) for sampled actions.
        teacher_logprobs: Per-token log p_teacher(a_t | prefix) on the same tokens.
        delta_scale: Positive constant; larger → sharper agreement requirement.

    Returns:
        Tensor of shape (T,) with values in (0, 1].
    """
    diff = (student_logprobs - teacher_logprobs).abs()
    return torch.exp(-delta_scale * diff).clamp(1e-8, 1.0)


def terminal_potential_from_reward(
    total_reward: float,
    reward_scale: float,
) -> float:
    """Map scalar trajectory return to a terminal factor R in (0, 1].

    Large positive rewards → R → 1; large negative → R → 0.
    """
    if reward_scale <= 0:
        reward_scale = 1.0
    r = float(torch.sigmoid(torch.tensor(reward_scale * float(total_reward))).item())
    return max(r, 1e-8)


def forward_log_messages(emission: torch.Tensor) -> torch.Tensor:
    """Chain forward: log α[t] = sum_{k<=t} log e_k."""
    log_e = _safe_log(emission)
    return torch.cumsum(log_e, dim=0)


def backward_log_messages(emission: torch.Tensor, terminal_r: float) -> torch.Tensor:
    """Chain backward with terminal R at the end of the sequence.

    log β[T-1] = log R
    log β[t] = log β[t+1] + log e[t+1]  for t = T-2 .. 0

    So β[t] aggregates emissions *after* t and the terminal potential (HMM-style).
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
    """Per-token KL multipliers from z-scored log α vs log β contrast.

    When z(log β) - z(log α) is large, backward (global / terminal-facing) mass
    dominates at that position → lower KL weight. When the opposite holds,
    weight increases to pull the student toward the teacher.

    Args:
        log_alpha: Forward log messages, shape (T,).
        log_beta: Backward log messages, shape (T,).
        gate_temperature: Sharpness of the sigmoid gate (same role as WGHD gate).

    Returns:
        KL weights in (0, 1), shape (T,).
    """
    if log_alpha.numel() == 0:
        return log_alpha
    z_a = (log_alpha - log_alpha.mean()) / (log_alpha.std(unbiased=False).clamp(min=1e-6))
    z_b = (log_beta - log_beta.mean()) / (log_beta.std(unbiased=False).clamp(min=1e-6))
    # High z_b relative to z_a → lower KL (trust global signal more).
    x = gate_temperature * (z_b - z_a)
    return torch.sigmoid(x)


def compute_fb_kl_weights_and_messages(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    total_reward: float,
    *,
    emission_delta_scale: float,
    terminal_reward_scale: float,
    gate_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-token KL weights and diagnostic log α, log β.

    Args:
        student_logprobs: Shape (T,).
        teacher_logprobs: Shape (T,), aligned with student on sampled tokens.
        total_reward: Scalar trajectory return (sum of step rewards + group reward).
        emission_delta_scale: Passed to :func:`local_emission_from_logprob_diff`.
        terminal_reward_scale: Passed to :func:`terminal_potential_from_reward`.
        gate_temperature: Sigmoid sharpness for KL weights.

    Returns:
        (kl_weights, log_alpha, log_beta, emission) each shape (T,) except scalars N/A.
    """
    if student_logprobs.shape[0] == 0:
        z = student_logprobs
        return z, z, z, z
    emission = local_emission_from_logprob_diff(
        student_logprobs, teacher_logprobs, emission_delta_scale
    )
    r = terminal_potential_from_reward(total_reward, terminal_reward_scale)
    log_alpha = forward_log_messages(emission)
    log_beta = backward_log_messages(emission, r)
    kl_w = kl_weights_from_fb_tension(log_alpha, log_beta, gate_temperature)
    return kl_w, log_alpha, log_beta, emission
