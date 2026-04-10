"""
Waypoint-Guided Hybrid Distillation (WGHD) for reasoning and code tasks.

This recipe implements WGHD, an extension of on-policy distillation that uses
Forward-Backward inspired adaptive KL gating. Instead of applying a uniform
KL penalty across all tokens, WGHD dynamically adjusts the KL weight per
segment based on how well the student's generation aligns with the teacher's
logical waypoints (e.g., reasoning steps in math, function boundaries in code).

Key features:
  - Adaptive KL gating: Low KL when student is on-track (exploration mode),
    high KL when student diverges (correction mode)
  - Process rewards: Bonus reward for matching each teacher waypoint
  - Segment-level granularity: Between pure token-level and sequence-level

Example usage:
    # Math (DeepMath) with WGHD
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=deepmath \\
        learning_rate=1e-4 \\
        groups_per_batch=1024 \\
        lora_rank=128 \\
        wandb_project=cookbook_wghd

    # With custom waypoint settings
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=deepmath \\
        similarity_threshold=0.4 \\
        gate_temperature=8.0 \\
        step_reward=0.15 \\
        wandb_project=cookbook_wghd

    # Code tasks
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=tulu3 \\
        task_type=code \\
        wandb_project=cookbook_wghd

    # Disable waypoint gating (falls back to standard on-policy distillation)
    python -m tinker_cookbook.recipes.distillation.wghd_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        dataset=deepmath \\
        waypoint_enabled=false \\
        wandb_project=cookbook_wghd
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

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for WGHD distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "deepmath"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    num_substeps: int = 1

    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # WGHD-specific: waypoint gating parameters
    waypoint_enabled: bool = True
    task_type: str = "math"
    similarity_threshold: float = 0.5
    gate_temperature: float = 5.0
    step_reward: float = 0.1
    min_gap_chars: int = 50
    waypoint_store_path: str | None = None

    # Logging configuration
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

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"wghd-{cli_config.dataset}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"t{cli_config.similarity_threshold}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = f"/tmp/tinker-examples/wghd/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = Path(log_path).name

    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name=cli_config.dataset,
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
