from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    pad_id = tokenizer.pad_token_id
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    full_sequences = [p + o for p, o in zip(prompt_ids, output_ids, strict=True)]
    max_len = max(len(seq) - 1 for seq in full_sequences)

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []
    for p, o, seq in zip(prompt_ids, output_ids, full_sequences, strict=True):
        seq_len = len(seq) - 1
        pad_n = max_len - seq_len
        input_ids.append(seq[:-1] + [pad_id] * pad_n)
        labels.append(seq[1:] + [pad_id] * pad_n)
        # Labels are shifted by one, so the first len(p)-1 label positions still
        # correspond to prompt tokens; the next len(o) are the response.
        response_mask.append([False] * (len(p) - 1) + [True] * len(o) + [False] * pad_n)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out: dict[str, Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)
    return out


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    raw_list: list[float] = []
    fmt_list: list[float] = []
    ans_list: list[float] = []
    for response, gt in zip(rollout_responses, repeated_ground_truths, strict=True):
        info = reward_fn(response, gt)
        raw_list.append(float(info["reward"]))
        fmt_list.append(float(info.get("format_reward", 0.0)))
        ans_list.append(float(info.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(raw_list, dtype=torch.float32)
    grouped = raw_rewards.view(-1, group_size)
    centered = grouped - grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        stds = grouped.std(dim=1, keepdim=True, unbiased=False)
        normalized = centered / (stds + advantage_eps)
    else:
        normalized = centered
    advantages = normalized.reshape(-1)

    metadata: dict[str, float] = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()),
        "reward_max": float(raw_rewards.max().item()),
        "reward_min": float(raw_rewards.min().item()),
        "format_reward_mean": float(sum(fmt_list) / len(fmt_list)) if fmt_list else 0.0,
        "answer_reward_mean": float(sum(ans_list) / len(ans_list)) if ans_list else 0.0,
        "group_std_mean": float(grouped.std(dim=1, unbiased=False).mean().item()),
    }
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    adv = advantages.expand_as(policy_log_probs)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    unclipped_obj = ratio * adv
    clipped_obj = clipped * adv
    per_token_loss = -torch.minimum(unclipped_obj, clipped_obj)

    out_of_range = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    metadata = {
        "clip_mask": out_of_range,
        "clip_fraction": out_of_range.float().mean().detach(),
        "ratio_mean": ratio.mean().detach(),
    }
    return per_token_loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, loss_metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask = response_mask.to(per_token_loss.dtype)
    masked = per_token_loss * mask
    # Masked mean over the sequence dim, then mean over batch.
    per_example = masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    batch_loss = per_example.mean()
    loss = batch_loss / gradient_accumulation_steps
    loss.backward()

    metadata = dict(loss_metadata)
    metadata["loss"] = loss.detach()
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
    response_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Create serializable generation logs for debugging training runs.

    Returns a dict with ``records`` (one entry per prompt) and ``summary`` with
    average response length overall and split by correctness, matching the six
    items §3.3.2 asks us to log.
    """
    records: list[dict[str, Any]] = []
    all_lengths: list[int] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []

    for i, (prompt, response, gt, info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        length = int(response_lengths[i]) if response_lengths is not None else len(response)
        avg_entropy = float(token_entropies[i]) if token_entropies is not None else None
        is_correct = info.get("answer_reward", 0.0) == 1.0

        records.append(
            {
                "prompt": prompt,
                "response": response,
                "ground_truth": gt,
                "format_reward": info.get("format_reward", 0.0),
                "answer_reward": info.get("answer_reward", 0.0),
                "reward": info.get("reward", 0.0),
                "avg_token_entropy": avg_entropy,
                "response_length": length,
                "correct": is_correct,
            }
        )
        all_lengths.append(length)
        (correct_lengths if is_correct else incorrect_lengths).append(length)

    def _mean(xs: list[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    summary = {
        "n": len(records),
        "n_correct": len(correct_lengths),
        "n_incorrect": len(incorrect_lengths),
        "avg_response_length": _mean(all_lengths),
        "avg_correct_length": _mean(correct_lengths),
        "avg_incorrect_length": _mean(incorrect_lengths),
    }
    return {"records": records, "summary": summary}


