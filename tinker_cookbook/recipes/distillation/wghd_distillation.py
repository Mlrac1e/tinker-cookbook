"""
Waypoint-Guided Hybrid Distillation (WGHD) for reasoning and code tasks.

This recipe implements WGHD, an extension of on-policy distillation. By default
it uses **latent-state token-chain forward-backward messages** (see
``tinker_cookbook.distillation.fb_chain``) to set per-token KL weights from
local teacher-student agreement (emission), waypoint-induced transitions, and
a terminal potential derived from trajectory return. Optional
``kl_weight_mode=cosine`` restores the legacy segment-wise cosine gating.

Datasets:

* ``gsm8k`` / ``math500`` / ``hendrycks_math`` — *verifiable* math envs from
  ``tinker_cookbook.recipes.math_rl`` (correctness rewards via boxed answer
  grading). **This is the recommended pilot dataset** because the trajectory
  return drives the FB terminal factor.
* ``deepmath`` / ``tulu3`` — prompt-only datasets (no env reward); the FB
  terminal factor will be uninformative (``R = 0.5`` for every trajectory)
  unless you wire in a separate process reward. Useful for emission-only
  ablations.

Example usage::

    # GSM8K pilot with default latent FB settings (paper main configuration).
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen2.5-1.5B \\
        teacher_model=Qwen/Qwen2.5-7B-Instruct \\
        dataset=gsm8k \\
        task_type=math \\
        learning_rate=1e-4 \\
        groups_per_batch=128 \\
        group_size=4 \\
        max_tokens=1024 \\
        wandb_project=wghd_gsm8k_pilot

    # Ablation: legacy single-state FB (z-scored) without latent posterior.
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen2.5-1.5B \\
        teacher_model=Qwen/Qwen2.5-7B-Instruct \\
        dataset=gsm8k \\
        fb_use_latent=false \\
        wandb_project=wghd_gsm8k_pilot

    # Ablation: drop waypoint transitions (uniform stationary transition).
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen2.5-1.5B \\
        teacher_model=Qwen/Qwen2.5-7B-Instruct \\
        dataset=gsm8k \\
        fb_use_waypoint_transition=false \\
        wandb_project=wghd_gsm8k_pilot

    # Ablation: multiplicative R injection on top of the terminal factor.
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen2.5-1.5B \\
        teacher_model=Qwen/Qwen2.5-7B-Instruct \\
        dataset=gsm8k \\
        fb_gate_exponent_reward=1.0 \\
        wandb_project=wghd_gsm8k_pilot

    # Disable waypoint gating altogether (falls back to vanilla on-policy KL).
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen2.5-1.5B \\
        teacher_model=Qwen/Qwen2.5-7B-Instruct \\
        dataset=gsm8k \\
        waypoint_enabled=false \\
        wandb_project=wghd_gsm8k_pilot
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_wghd
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)
from tinker_cookbook.distillation.train_wghd import WaypointConfig
from tinker_cookbook.recipes.math_rl.math_env import (
    Gsm8kDatasetBuilder,
    MathDatasetBuilder,
)
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)

#: Datasets that use ``math_rl.MathEnv`` and therefore expose a verifiable
#: trajectory reward (correct/incorrect via boxed-answer grading). These are
#: required to exercise the FB terminal factor.
_VERIFIABLE_MATH_DATASETS = {"gsm8k", "math", "hendrycks_math"}


def _build_dataset_builder(
    dataset_name: str,
    *,
    groups_per_batch: int,
    group_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
) -> RLDatasetBuilder:
    """Pick a dataset builder for distillation based on ``dataset_name``.

    Verifiable math datasets (``gsm8k`` / ``math``) carry per-trajectory
    correctness rewards needed by the FB terminal factor. Prompt-only datasets
    (``deepmath`` / ``tulu3``) provide no env reward; they fall back to a
    constant ``R = 0.5`` and only the FB emission and waypoint transitions
    will inform the per-token weights.
    """
    if dataset_name == "gsm8k":
        return Gsm8kDatasetBuilder(
            batch_size=groups_per_batch,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
        )
    if dataset_name in {"math", "hendrycks_math"}:
        return MathDatasetBuilder(
            batch_size=groups_per_batch,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
            group_size=group_size,
        )
    if dataset_name in {"deepmath", "tulu3"}:
        return PromptOnlyDatasetBuilder(
            dataset_name=dataset_name,
            groups_per_batch=groups_per_batch,
            group_size=group_size,
            model_name_for_tokenizer=model_name_for_tokenizer,
            renderer_name=renderer_name,
        )
    raise ValueError(
        f"Unknown dataset {dataset_name!r}. "
        f"Supported: gsm8k, math, hendrycks_math, deepmath, tulu3."
    )


@chz.chz
class CLIConfig:
    """Command-line configuration for WGHD distillation."""

    # --- Model configuration ---
    model_name: str = "Qwen/Qwen2.5-1.5B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # --- Teacher configuration ---
    teacher_model: str = "Qwen/Qwen2.5-7B-Instruct"
    teacher_checkpoint: str | None = None

    # --- Dataset configuration ---
    #: One of ``gsm8k``, ``math``, ``hendrycks_math``, ``deepmath``, ``tulu3``.
    #: Use a verifiable math dataset to exercise the FB terminal reward factor.
    dataset: str = "gsm8k"

    # --- Training hyperparameters ---
    group_size: int = 4
    groups_per_batch: int = 128
    learning_rate: float = 1e-4
    max_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    num_substeps: int = 1

    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # --- WGHD: gating mode ---
    waypoint_enabled: bool = True
    #: ``fb_chain`` (default) = latent-state forward-backward;
    #: ``cosine`` = legacy waypoint-similarity heuristic.
    kl_weight_mode: str = "fb_chain"
    task_type: str = "math"
    similarity_threshold: float = 0.5
    gate_temperature: float = 5.0
    step_reward: float = 0.1
    min_gap_chars: int = 50
    waypoint_store_path: str | None = None

    # --- Latent FB hyperparameters (paper-method knobs) ---
    fb_emission_delta_scale: float = 1.0
    fb_terminal_reward_scale: float = 1.0
    #: Latent track-state cardinality. ``2`` = on/off (paper default).
    fb_num_states: int = 2
    #: Constant emission level for the off-track latent state.
    fb_emission_off_level: float = 0.3
    #: Per-step state flip probability at non-boundary positions.
    fb_transition_flip_prob: float = 0.05
    #: Mixing weight toward uniform transition at waypoint boundaries.
    fb_boundary_reset_prob: float = 0.5
    #: Exponent on ``gamma_t(on-track)`` in the per-token KL weight.
    fb_gate_exponent_gamma: float = 1.0
    #: Exponent on the multiplicative ``R`` injection on the per-token weight.
    #: ``0.0`` keeps the outcome influence purely through the terminal beta.
    fb_gate_exponent_reward: float = 0.0
    #: Initial probability of being on-track.
    fb_prior_on_track: float = 0.5
    #: If ``False``, ignore waypoint boundaries (stationary transition).
    fb_use_waypoint_transition: bool = True
    #: If ``False``, fall back to the legacy single-state alpha/beta gate
    #: (ablation; useful to measure the contribution of latent FB).
    fb_use_latent: bool = True

    # --- Logging configuration ---
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    eval_every: int = 20
    save_every: int = 20

    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run WGHD training."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    if cli_config.dataset not in _VERIFIABLE_MATH_DATASETS:
        logger.warning(
            "Dataset %r has no env reward; the FB terminal factor will be "
            "constant (R = 0.5). Use 'gsm8k' or 'math' for a meaningful "
            "outcome signal.",
            cli_config.dataset,
        )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"wghd-{cli_config.dataset}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"K{cli_config.fb_num_states}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = f"/tmp/tinker-examples/wghd/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = Path(log_path).name

    dataset_builder = _build_dataset_builder(
        cli_config.dataset,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )

    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    waypoint_config = WaypointConfig(
        enabled=cli_config.waypoint_enabled,
        task_type=cli_config.task_type,
        similarity_threshold=cli_config.similarity_threshold,
        gate_temperature=cli_config.gate_temperature,
        step_reward=cli_config.step_reward,
        min_gap_chars=cli_config.min_gap_chars,
        waypoint_store_path=cli_config.waypoint_store_path,
        kl_weight_mode=cli_config.kl_weight_mode,
        fb_emission_delta_scale=cli_config.fb_emission_delta_scale,
        fb_terminal_reward_scale=cli_config.fb_terminal_reward_scale,
        fb_num_states=cli_config.fb_num_states,
        fb_emission_off_level=cli_config.fb_emission_off_level,
        fb_transition_flip_prob=cli_config.fb_transition_flip_prob,
        fb_boundary_reset_prob=cli_config.fb_boundary_reset_prob,
        fb_gate_exponent_gamma=cli_config.fb_gate_exponent_gamma,
        fb_gate_exponent_reward=cli_config.fb_gate_exponent_reward,
        fb_prior_on_track=cli_config.fb_prior_on_track,
        fb_use_waypoint_transition=cli_config.fb_use_waypoint_transition,
        fb_use_latent=cli_config.fb_use_latent,
    )

    config = train_wghd.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        waypoint_config=waypoint_config,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await train_wghd.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
