"""
Waypoint extraction and similarity computation for Waypoint-Guided Hybrid Distillation (WGHD).

This module implements the "offline waypoint mining" phase: given a teacher model's
correct trajectories on structured tasks (math, code), extract logit-based feature
vectors at logical breakpoints ("waypoints"). At training time, the student's features
are compared against these waypoints via cosine similarity to dynamically gate
the KL penalty.

Waypoints are identified at logical boundaries in the generated text:
- Math: double newlines, "Step N:", numbered list items
- Code: function definitions, loop constructs, return statements

The core abstraction is:
  WaypointFeature: a per-position feature vector (top-K logits from teacher)
  WaypointSequence: ordered list of waypoint features for one problem
  WaypointStore: in-memory mapping from problem prompts to their waypoint sequences
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# --- Waypoint boundary detection ---

MATH_BOUNDARY_PATTERN = re.compile(
    r"(?:"
    r"\n\n"                     # double newline
    r"|(?:Step\s+\d+)"         # "Step N"
    r"|(?:^\d+[\.\)]\s)"       # numbered list "1. " or "1) "
    r"|(?:\\boxed\{)"          # boxed answer start
    r"|(?:Therefore[,:])"      # conclusion markers
    r"|(?:Thus[,:])"
    r"|(?:Hence[,:])"
    r"|(?:So[,:])"
    r")",
    re.MULTILINE,
)

CODE_BOUNDARY_PATTERN = re.compile(
    r"(?:"
    r"\ndef\s"                  # function definition
    r"|for\s+\w+\s+in\s"       # for loop
    r"|while\s"                 # while loop
    r"|if\s"                    # if statement
    r"|return\s"               # return statement
    r"|class\s"                # class definition
    r"|```"                    # code block boundary
    r")",
    re.MULTILINE,
)


def detect_boundary_positions(
    text: str,
    task_type: str = "math",
    min_gap_chars: int = 50,
) -> list[int]:
    """Find character positions in `text` that correspond to logical waypoint boundaries.

    Args:
        text: The full generated text from the teacher.
        task_type: "math" or "code" -- selects the boundary regex.
        min_gap_chars: Minimum character distance between consecutive boundaries
            to avoid excessively dense waypoints.

    Returns:
        Sorted list of character offsets where boundaries were detected.
    """
    pattern = MATH_BOUNDARY_PATTERN if task_type == "math" else CODE_BOUNDARY_PATTERN
    raw_positions = sorted({m.start() for m in pattern.finditer(text)})

    filtered: list[int] = []
    for pos in raw_positions:
        if not filtered or (pos - filtered[-1]) >= min_gap_chars:
            filtered.append(pos)
    return filtered


def char_offset_to_token_index(
    text: str,
    char_offset: int,
    token_offsets: list[int],
) -> int:
    """Map a character offset to the nearest token index.

    Args:
        text: Full text string.
        char_offset: Character position in text.
        token_offsets: Character start position for each token (same length as token list).

    Returns:
        Index of the token whose start position is closest to (and <= ) char_offset.
    """
    best_idx = 0
    for i, tok_start in enumerate(token_offsets):
        if tok_start <= char_offset:
            best_idx = i
        else:
            break
    return best_idx


# --- Feature vectors ---

@dataclass
class WaypointFeature:
    """A feature vector extracted at a single waypoint position.

    We store the teacher's top-K log-probabilities at that token position
    as a sparse representation. For cosine similarity we use these as a
    fixed-dimensional vector (indices map to vocab positions).

    Attributes:
        token_index: The token position in the teacher's output sequence.
        top_k_indices: Vocab indices of the top-K tokens.
        top_k_logprobs: Corresponding log-probabilities.
        char_offset: Original character offset in the decoded text.
    """
    token_index: int
    top_k_indices: list[int]
    top_k_logprobs: list[float]
    char_offset: int = 0

    def to_dense_vector(self, vocab_size: int = 0) -> torch.Tensor:
        """Convert to a dense probability vector (softmax of top-K logprobs).

        When vocab_size=0, returns a vector of length len(top_k_indices)
        containing the softmaxed logprobs (useful for cosine similarity
        without materializing a full vocab-sized vector).
        """
        logprobs = torch.tensor(self.top_k_logprobs, dtype=torch.float32)
        probs = torch.softmax(logprobs, dim=0)
        return probs

    def to_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "top_k_indices": self.top_k_indices,
            "top_k_logprobs": self.top_k_logprobs,
            "char_offset": self.char_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WaypointFeature:
        return cls(
            token_index=d["token_index"],
            top_k_indices=d["top_k_indices"],
            top_k_logprobs=d["top_k_logprobs"],
            char_offset=d.get("char_offset", 0),
        )


@dataclass
class WaypointSequence:
    """An ordered sequence of waypoints for a single problem/trajectory.

    Attributes:
        prompt: The original problem prompt.
        waypoints: Ordered list of waypoint features along the trajectory.
        task_type: "math" or "code".
        full_response: The teacher's full response text (for debugging).
    """
    prompt: str
    waypoints: list[WaypointFeature]
    task_type: str = "math"
    full_response: str = ""

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "waypoints": [w.to_dict() for w in self.waypoints],
            "task_type": self.task_type,
            "full_response": self.full_response,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WaypointSequence:
        return cls(
            prompt=d["prompt"],
            waypoints=[WaypointFeature.from_dict(w) for w in d["waypoints"]],
            task_type=d.get("task_type", "math"),
            full_response=d.get("full_response", ""),
        )


class WaypointStore:
    """In-memory store mapping problem prompts to their waypoint sequences.

    Supports serialization to/from JSONL for persistence.
    """

    def __init__(self) -> None:
        self._store: dict[str, WaypointSequence] = {}

    def add(self, seq: WaypointSequence) -> None:
        self._store[seq.prompt] = seq

    def get(self, prompt: str) -> WaypointSequence | None:
        return self._store.get(prompt)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._store

    def save(self, path: str | Path) -> None:
        """Persist to a JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for seq in self._store.values():
                f.write(json.dumps(seq.to_dict()) + "\n")
        logger.info(f"Saved {len(self._store)} waypoint sequences to {path}")

    @classmethod
    def load(cls, path: str | Path) -> WaypointStore:
        """Load from a JSONL file."""
        store = cls()
        path = Path(path)
        with open(path) as f:
            for line in f:
                if line.strip():
                    seq = WaypointSequence.from_dict(json.loads(line))
                    store.add(seq)
        logger.info(f"Loaded {len(store)} waypoint sequences from {path}")
        return store


# --- Similarity computation ---

def compute_waypoint_similarity(
    student_logprobs: torch.Tensor,
    waypoint: WaypointFeature,
) -> float:
    """Compute cosine similarity between student's logprob distribution and a waypoint.

    We compare the student's log-probabilities at the same vocab positions as
    the teacher's top-K, using cosine similarity on the softmaxed distributions.

    Args:
        student_logprobs: Full logprob vector from the student at a given position.
            Shape: (vocab_size,) or a 1D tensor of logprobs at the top-K indices.
        waypoint: The teacher's waypoint feature to compare against.

    Returns:
        Cosine similarity in [-1, 1].
    """
    teacher_vec = waypoint.to_dense_vector()
    indices = torch.tensor(waypoint.top_k_indices, dtype=torch.long)

    if student_logprobs.dim() == 1 and student_logprobs.shape[0] > len(indices):
        student_at_positions = student_logprobs[indices]
    else:
        student_at_positions = student_logprobs[: len(indices)]

    student_probs = torch.softmax(student_at_positions, dim=0)

    cos_sim = torch.nn.functional.cosine_similarity(
        student_probs.unsqueeze(0),
        teacher_vec.unsqueeze(0),
    )
    return float(cos_sim.item())


def compute_adaptive_kl_weight(
    similarity: float,
    threshold: float = 0.5,
    temperature: float = 5.0,
) -> float:
    """Compute a smooth KL weight from waypoint similarity using a sigmoid gate.

    When similarity is high (student is on track), KL weight is low (allow exploration).
    When similarity is low (student is off track), KL weight is high (enforce teacher KD).

    The weight is computed as: sigmoid(temperature * (threshold - similarity))

    Args:
        similarity: Cosine similarity between student and teacher waypoint, in [-1, 1].
        threshold: Similarity level below which KL enforcement kicks in.
        temperature: Controls the sharpness of the sigmoid transition.

    Returns:
        KL weight in [0, 1].
    """
    x = temperature * (threshold - similarity)
    return float(torch.sigmoid(torch.tensor(x)).item())


# --- Segment-level waypoint matching ---

def find_active_waypoint_index(
    current_token_position: int,
    waypoints: list[WaypointFeature],
) -> int:
    """Find which waypoint the student should be targeting at the current position.

    Returns the index of the next waypoint the student hasn't passed yet.
    If all waypoints have been passed, returns the last waypoint index.

    Args:
        current_token_position: The student's current token generation position.
        waypoints: Ordered list of waypoint features.

    Returns:
        Index into the waypoints list.
    """
    for i, wp in enumerate(waypoints):
        if wp.token_index > current_token_position:
            return i
    return len(waypoints) - 1


def compute_segment_kl_weights(
    student_token_count: int,
    waypoints: list[WaypointFeature],
    segment_similarities: list[float],
    threshold: float = 0.5,
    temperature: float = 5.0,
    step_reward: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Compute per-token KL weights for an entire sequence based on waypoint matching.

    For each token position, determines the active waypoint target and computes
    an adaptive KL weight based on the similarity at that segment boundary.
    Also computes the total process reward from successfully matched waypoints.

    Args:
        student_token_count: Number of tokens in the student's output.
        waypoints: Ordered teacher waypoint features.
        segment_similarities: Cosine similarities at each waypoint boundary
            (one per waypoint).
        threshold: Similarity threshold for the KL gate.
        temperature: Sigmoid temperature for the KL gate.
        step_reward: Bonus reward for each waypoint matched above threshold.

    Returns:
        Tuple of (kl_weights, process_reward):
        - kl_weights: shape (student_token_count,) with per-token KL weights
        - process_reward: scalar bonus reward from waypoint matches
    """
    kl_weights = torch.ones(student_token_count, dtype=torch.float32)
    total_process_reward = 0.0

    if not waypoints or not segment_similarities:
        return kl_weights, total_process_reward

    wp_boundaries = [wp.token_index for wp in waypoints]

    for t in range(student_token_count):
        active_idx = find_active_waypoint_index(t, waypoints)
        if active_idx < len(segment_similarities):
            sim = segment_similarities[active_idx]
            kl_weights[t] = compute_adaptive_kl_weight(sim, threshold, temperature)

    for sim in segment_similarities:
        if sim >= threshold:
            total_process_reward += step_reward

    return kl_weights, total_process_reward
