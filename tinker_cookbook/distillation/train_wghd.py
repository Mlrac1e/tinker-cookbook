"""
Waypoint-Guided Hybrid Distillation (WGHD) training engine.

Extends on-policy distillation with Forward-Backward inspired adaptive KL gating.
Instead of applying a uniform KL penalty across all tokens, WGHD:

1. Identifies logical waypoints in the teacher's correct trajectories
2. During on-policy training, compares the student's generation against
   these waypoints using cosine similarity on logprob distributions
3. Dynamically adjusts KL penalty per segment:
   - High similarity → low KL weight (allow student exploration)
   - Low similarity → high KL weight (enforce teacher guidance)
4. Awards process rewards for successfully matching waypoints

This provides a middle ground between pure token-level KL (too myopic) and
pure sequence-level RL (too sparse), implementing the "look forward and backward"
philosophy described in the Forward-Backward representation theory.

Usage:
    from tinker_cookbook.distillation import train_wghd
    config = train_wghd.Config(...)
    await train_wghd.main(config)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import chz
import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)
from tinker_cookbook.distillation.waypoint import (
    WaypointFeature,
    WaypointSequence,
    WaypointStore,
    compute_adaptive_kl_weight,
    compute_segment_kl_weights,
    compute_waypoint_similarity,
    detect_boundary_positions,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.misc_utils import iteration_dir, safezip

logger = logging.getLogger(__name__)


@chz.chz
class WaypointConfig:
    """Configuration for waypoint-guided KL gating."""

    enabled: bool = True
    task_type: str = "math"
    similarity_threshold: float = 0.5
    gate_temperature: float = 5.0
    step_reward: float = 0.1
    min_gap_chars: int = 50
    waypoint_store_path: str | None = None


@trace.scope
async def incorporate_kl_penalty_with_waypoints(
    data_D: list[tinker.Datum],
    teacher_clients_D: list[tinker.SamplingClient],
    dataset_indices_D: list[int],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    waypoint_config: WaypointConfig,
    waypoint_store: WaypointStore | None = None,
    tokenizer: Tokenizer | None = None,
) -> dict[str, float]:
    """Compute reverse KL with adaptive per-token weighting based on waypoint similarity.

    Extends the standard on-policy KL penalty with waypoint-guided gating:
    - Segments where the student aligns with teacher waypoints get reduced KL
    - Segments where the student diverges get amplified KL
    - Process rewards are added to advantages for waypoint matches

    Args:
        data_D: List of datums to compute KL for.
        teacher_clients_D: List of teacher sampling clients, one per datum.
        dataset_indices_D: List of dataset indices, one per datum.
        kl_penalty_coef: Base coefficient for KL penalty.
        kl_discount_factor: Discount factor for future KL.
        waypoint_config: Configuration for waypoint gating behavior.
        waypoint_store: Pre-computed waypoint features for teacher trajectories.
        tokenizer: Tokenizer for decoding student outputs (needed for waypoint matching).
    """
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]

    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(sequence_input)
            for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
        ]
    )

    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = [
        (sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask
        for teacher_logprobs, sampled_logprobs, mask in safezip(
            teacher_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]

    per_dataset_kl: dict[int, tuple[float, float]] = {}
    total_process_reward = 0.0
    waypoints_matched = 0
    waypoints_total = 0

    for i, datum in enumerate(data_D):
        kl_weight_per_token = torch.ones_like(float_masks[i])
        process_reward = 0.0

        if (
            waypoint_config.enabled
            and waypoint_store is not None
            and tokenizer is not None
        ):
            kl_weight_per_token, process_reward = _compute_waypoint_gated_weights(
                datum=datum,
                teacher_logprobs=teacher_logprobs_D[i],
                tokenizer=tokenizer,
                waypoint_store=waypoint_store,
                waypoint_config=waypoint_config,
            )
            total_process_reward += process_reward
            if process_reward > 0:
                waypoints_matched += int(process_reward / waypoint_config.step_reward)
            waypoints_total += 1

        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i] * kl_weight_per_token
        if kl_discount_factor > 0:
            kl_advantages = discounted_future_sum_vectorized(kl_advantages, kl_discount_factor)

        current_advantages = datum.loss_fn_inputs["advantages"].to_torch()
        process_reward_tensor = torch.full_like(current_advantages, process_reward) * float_masks[i]
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            current_advantages + kl_advantages + process_reward_tensor
        )

        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = float_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / max(
        sum([mask.sum() for mask in float_masks]), 1.0
    )

    metrics: dict[str, float] = {
        "teacher_kl": float(avg_logp_diff),
        "wghd/process_reward_total": total_process_reward,
        "wghd/waypoints_matched": float(waypoints_matched),
        "wghd/waypoints_evaluated": float(waypoints_total),
    }
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    return metrics


def _compute_waypoint_gated_weights(
    datum: tinker.Datum,
    teacher_logprobs: list[float],
    tokenizer: Tokenizer,
    waypoint_store: WaypointStore,
    waypoint_config: WaypointConfig,
) -> tuple[torch.Tensor, float]:
    """Compute per-token KL weights using waypoint similarity for a single datum.

    This is the core FB-inspired gating mechanism. For each waypoint boundary
    in the teacher's trajectory, we measure how well the student's generation
    aligns with the teacher at that checkpoint. Segments that are on-track
    get reduced KL (exploration mode), while off-track segments get amplified
    KL (correction mode).

    Returns:
        Tuple of (kl_weights, process_reward) where kl_weights has the same
        length as the datum's mask tensor.
    """
    mask = datum.loss_fn_inputs["mask"].to_torch()
    seq_len = len(mask)
    kl_weights = torch.ones(seq_len, dtype=torch.float32)
    process_reward = 0.0

    target_tokens = datum.loss_fn_inputs["target_tokens"].data
    try:
        decoded_text = tokenizer.decode(target_tokens)
    except Exception:
        return kl_weights, process_reward

    boundary_chars = detect_boundary_positions(
        decoded_text,
        task_type=waypoint_config.task_type,
        min_gap_chars=waypoint_config.min_gap_chars,
    )

    if not boundary_chars:
        return kl_weights, process_reward

    teacher_logprobs_tensor = torch.tensor(teacher_logprobs[1:], dtype=torch.float32)
    if len(teacher_logprobs_tensor) != seq_len:
        teacher_logprobs_tensor = teacher_logprobs_tensor[:seq_len]
        if len(teacher_logprobs_tensor) < seq_len:
            pad = torch.zeros(seq_len - len(teacher_logprobs_tensor))
            teacher_logprobs_tensor = torch.cat([teacher_logprobs_tensor, pad])

    student_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()

    segment_boundaries = _estimate_token_boundaries(decoded_text, boundary_chars, seq_len)
    segment_similarities: list[float] = []

    for boundary_token_idx in segment_boundaries:
        if boundary_token_idx >= seq_len:
            boundary_token_idx = seq_len - 1
        if boundary_token_idx < 0:
            continue

        teacher_local = teacher_logprobs_tensor[
            max(0, boundary_token_idx - 2) : min(seq_len, boundary_token_idx + 3)
        ]
        student_local = student_logprobs[
            max(0, boundary_token_idx - 2) : min(seq_len, boundary_token_idx + 3)
        ]

        if len(teacher_local) > 0 and len(student_local) > 0:
            min_len = min(len(teacher_local), len(student_local))
            teacher_local = teacher_local[:min_len]
            student_local = student_local[:min_len]

            teacher_probs = torch.softmax(teacher_local, dim=0)
            student_probs = torch.softmax(student_local, dim=0)
            sim = float(
                torch.nn.functional.cosine_similarity(
                    teacher_probs.unsqueeze(0),
                    student_probs.unsqueeze(0),
                ).item()
            )
        else:
            sim = 0.0
        segment_similarities.append(sim)

    prev_boundary = 0
    for seg_idx, boundary_token_idx in enumerate(segment_boundaries):
        if seg_idx >= len(segment_similarities):
            break
        sim = segment_similarities[seg_idx]
        weight = compute_adaptive_kl_weight(
            sim,
            waypoint_config.similarity_threshold,
            waypoint_config.gate_temperature,
        )

        end_idx = min(boundary_token_idx + 1, seq_len)
        kl_weights[prev_boundary:end_idx] = weight
        prev_boundary = end_idx

        if sim >= waypoint_config.similarity_threshold:
            process_reward += waypoint_config.step_reward

    if prev_boundary < seq_len:
        if segment_similarities:
            last_weight = compute_adaptive_kl_weight(
                segment_similarities[-1],
                waypoint_config.similarity_threshold,
                waypoint_config.gate_temperature,
            )
            kl_weights[prev_boundary:] = last_weight

    return kl_weights, process_reward


def _estimate_token_boundaries(
    text: str,
    char_positions: list[int],
    total_tokens: int,
) -> list[int]:
    """Estimate token indices from character positions using a linear mapping.

    This is an approximation: we assume roughly uniform characters-per-token
    across the sequence. A more precise version would use the tokenizer's
    offset mapping, but this is much cheaper and sufficient for gating.
    """
    if not text or total_tokens == 0:
        return []
    chars_per_token = len(text) / total_tokens
    if chars_per_token <= 0:
        return []
    return [
        min(int(cp / chars_per_token), total_tokens - 1)
        for cp in char_positions
    ]


# --- Config and training loop ---


@chz.chz
class Config:
    """Configuration for WGHD training."""
    learning_rate: float
    dataset_configs: list[DistillationDatasetConfig]
    model_name: str
    renderer_name: str | None = None
    max_tokens: int
    temperature: float = 1.0
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    num_substeps: int = 1

    waypoint_config: WaypointConfig = chz.field(default_factory=WaypointConfig)

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    base_url: str | None = None
    enable_trace: bool = False
    span_chart_every: int = 0

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    max_steps: int | None = None


@trace.scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: list[int],
    teacher_clients: list[tinker.SamplingClient],
    kl_penalty_coef: float,
    kl_discount_factor: float,
    waypoint_config: WaypointConfig,
    waypoint_store: WaypointStore | None = None,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts trajectories into a minibatch with waypoint-guided KL penalty."""

    metrics: dict[str, Any] = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    async with trace.scope_span("assemble_training_data"):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    printed_datasets: set[int] = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    if kl_penalty_coef > 0:
        async with trace.scope_span("compute_kl_penalty_with_waypoints"):
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics = await incorporate_kl_penalty_with_waypoints(
                data_D,
                teacher_clients_D,
                dataset_indices_D,
                kl_penalty_coef,
                kl_discount_factor,
                waypoint_config=waypoint_config,
                waypoint_store=waypoint_store,
                tokenizer=tokenizer,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@trace.scope
async def do_train_step_and_get_sampling_client(
    config: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: list[int],
    teacher_clients: list[tinker.SamplingClient],
    waypoint_store: WaypointStore | None = None,
    store: TrainingRunStore | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    trace.update_scope_context({"step": i_batch})

    metrics: dict[str, Any] = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
        waypoint_config=config.waypoint_config,
        waypoint_store=waypoint_store,
    )
    metrics.update(prepare_minibatch_metrics)

    async with trace.scope_span("train"):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=config.learning_rate,
            num_substeps=config.num_substeps,
            loss_fn=config.loss_fn,
            loss_fn_config=config.loss_fn_config,
            metrics=metrics,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data_D,
        training_logprobs_D,
        config.log_path,
        config.save_every,
        config.compute_post_kl,
        store=store,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@trace.scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: list[tinker.SamplingClient],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    waypoint_store: WaypointStore | None = None,
):
    """Implements fully synchronous WGHD training with waypoint-guided KL."""

    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, config.log_path, config.save_every, store=ml_logger.store
    )

    log_path = Path(config.log_path)

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            if config.eval_every > 0 and i_batch % config.eval_every == 0:
                async with trace.scope_span("run_evals"):
                    for evaluator in evaluators:
                        eval_metrics = await evaluator(sampling_client)
                        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

            env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
            async with trace.scope_span("sample"):
                trajectory_groups_P = await asyncio.gather(
                    *[
                        asyncio.create_task(
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                temperature=config.temperature,
                                max_tokens=config.max_tokens,
                                do_remove_constant_reward_groups=False,
                            ),
                            name=f"sample_task_{i}",
                        )
                        for i, builder in enumerate(env_group_builders_P)
                    ],
                )
            trajectory_groups_P = [
                trajectory_group
                for trajectory_group in trajectory_groups_P
                if trajectory_group is not None
            ]

            sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                config,
                i_batch,
                training_client,
                service_client,
                tokenizer,
                env_group_builders_P,
                trajectory_groups_P,
                dataset_indices_P,
                teacher_clients,
                waypoint_store=waypoint_store,
                store=ml_logger.store,
            )

            metrics.update(train_step_metrics)

        metrics.update(window.get_timing_metrics())
        window.save_timing(i_batch, store=ml_logger.store)
        if config.span_chart_every > 0 and i_batch % config.span_chart_every == 0:
            iter_dir = iteration_dir(log_path, i_batch)
            if iter_dir is not None:
                iter_dir.mkdir(parents=True, exist_ok=True)
                trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
        ml_logger.log_metrics(metrics, step=i_batch)


@trace.scope
async def main(config: Config):
    """Main training loop for WGHD."""

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    store = ml_logger.store
    if config.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        trace.trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_batch = resume_info.batch
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=config.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, config.renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )

    tokenizer = training_client.get_tokenizer()

    # Load or create waypoint store
    waypoint_store: WaypointStore | None = None
    if config.waypoint_config.enabled:
        if config.waypoint_config.waypoint_store_path:
            try:
                waypoint_store = WaypointStore.load(config.waypoint_config.waypoint_store_path)
                logger.info(
                    f"Loaded waypoint store with {len(waypoint_store)} entries "
                    f"from {config.waypoint_config.waypoint_store_path}"
                )
            except FileNotFoundError:
                logger.warning(
                    f"Waypoint store not found at {config.waypoint_config.waypoint_store_path}. "
                    "Proceeding without pre-computed waypoints."
                )
                waypoint_store = WaypointStore()
        else:
            waypoint_store = WaypointStore()
            logger.info(
                "No waypoint store path specified. "
                "WGHD will compute waypoint boundaries on-the-fly from teacher logprobs."
            )

    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in config.evaluator_builders]

    for dataset_config in config.dataset_configs:
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=config.max_tokens))

        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=teacher_config.base_model)
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        teacher_clients.append(teacher_client)
        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    num_batches = (
        min(config.max_steps, num_batches) if config.max_steps is not None else num_batches
    )
    logger.info(f"Will train on {num_batches} batches (dataset has {num_batches})")
    logger.info(
        f"WGHD config: enabled={config.waypoint_config.enabled}, "
        f"threshold={config.waypoint_config.similarity_threshold}, "
        f"temperature={config.waypoint_config.gate_temperature}, "
        f"step_reward={config.waypoint_config.step_reward}"
    )

    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        config=config,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        waypoint_store=waypoint_store,
    )

    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=None,
            store=store,
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("WGHD training completed successfully")
