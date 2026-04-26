"""GRPO training loop for §3.5.

Usage:

    python -m alignment.train --output-dir ./grpo_run --n-grpo-steps 50

vLLM is used for fast rollouts; HuggingFace for training. Weights are synced
from the HF policy into vLLM's in-memory model after every optimizer step.
"""
from __future__ import annotations

# Force vLLM V0 so we can load policy weights into the running engine. The V1
# engine reorganizes its internals and the in-memory weight-load path no longer
# works the same way.
import os
os.environ.setdefault("VLLM_USE_V1", "0")

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from .drgrpo_grader import r1_zero_reward_fn
from .grpo import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    tokenize_prompt_and_output,
)
from .prompts import COT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_gsm8k(split: str, max_examples: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[dict[str, Any]] = []
    for row in ds:
        ans = row["answer"]
        gt = ans.split("####")[-1].strip().replace(",", "") if "####" in ans else ans.strip()
        examples.append({"question": row["question"], "ground_truth": gt})
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def _init_vllm(model_name: str, gpu_memory_utilization: float, seed: int):
    """Init vLLM in the same process as the policy.

    We patch `_assert_memory_footprint_increased_during_profiling` because we
    cohabit with the HF model on the same GPU and that assertion can fire.
    """
    from vllm import LLM

    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with profiling_patch:
        return LLM(
            model=model_name,
            dtype="bfloat16",
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
        )


def _load_policy_into_vllm(policy, llm) -> None:
    """Sync HF policy weights into vLLM's in-memory model (V0 engine path)."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def _rollout(llm, prompts, group_size, max_tokens, min_tokens, temperature):
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,
    )
    outputs = llm.generate(list(prompts), params, use_tqdm=False)
    flat: list[str] = []
    for out in outputs:
        for completion in out.outputs:
            flat.append(completion.text)
    return flat


def _evaluate(llm, examples, max_tokens, temperature) -> dict[str, float]:
    from vllm import SamplingParams

    prompts = [COT_PROMPT_TEMPLATE.format(question=ex["question"]) for ex in examples]
    gts = [ex["ground_truth"] for ex in examples]
    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1,
    )
    outputs = llm.generate(prompts, params, use_tqdm=False)
    n_format = 0
    n_answer = 0
    for out, gt in zip(outputs, gts, strict=True):
        info = r1_zero_reward_fn(out.outputs[0].text, gt)
        n_format += int(info["format_reward"] == 1.0)
        n_answer += int(info["answer_reward"] == 1.0)
    n = len(prompts)
    return {
        "n": n,
        "format_accuracy": n_format / n if n else 0.0,
        "answer_accuracy": n_answer / n if n else 0.0,
    }


def _compute_old_log_probs(policy, input_ids, labels, micro_size, device):
    """Score every (input_ids, labels) pair under the current policy in chunks."""
    chunks: list[torch.Tensor] = []
    policy.eval()
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], micro_size):
            ids = input_ids[i : i + micro_size].to(device)
            lab = labels[i : i + micro_size].to(device)
            out = get_response_log_probs(policy, ids, lab, return_token_entropy=False)
            chunks.append(out["log_probs"].detach().to("cpu"))
    policy.train()
    return torch.cat(chunks, dim=0)


def train_grpo(
    output_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    n_grpo_steps: int = 50,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    grad_clip: float = 1.0,
    eval_every: int = 5,
    eval_size: int = 256,
    seed: int = 0,
    save_final: bool = True,
    vllm_gpu_memory_utilization: float = 0.3,
    attn_impl: str = "flash_attention_2",
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    val_path = output_dir / "validation.jsonl"
    rollouts_path = output_dir / "rollouts.jsonl"

    _set_seed(seed)

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_name} on {device} (attn={attn_impl})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    print(f"initializing vLLM (gpu_mem_util={vllm_gpu_memory_utilization})...")
    llm = _init_vllm(model_name, vllm_gpu_memory_utilization, seed)

    train_examples = _load_gsm8k("train")
    val_examples = _load_gsm8k("test")[:eval_size]
    print(f"train={len(train_examples)} val={len(val_examples)}")

    rng = random.Random(seed)
    metrics_log: list[dict[str, Any]] = []
    val_log: list[dict[str, Any]] = []

    print(
        f"GRPO: steps={n_grpo_steps} rollout_batch={rollout_batch_size} "
        f"group={group_size} train_batch={train_batch_size} "
        f"micro={micro_train_batch_size} accum={gradient_accumulation_steps} "
        f"normalize_by_std={normalize_by_std}"
    )

    for step in range(n_grpo_steps):
        t0 = time.time()

        sampled = rng.sample(train_examples, n_prompts_per_rollout_batch)
        prompts = [COT_PROMPT_TEMPLATE.format(question=ex["question"]) for ex in sampled]
        gts = [ex["ground_truth"] for ex in sampled]
        repeated_gts = [gt for gt in gts for _ in range(group_size)]
        repeated_prompts = [p for p in prompts for _ in range(group_size)]

        _load_policy_into_vllm(policy, llm)
        responses = _rollout(
            llm,
            prompts,
            group_size=group_size,
            max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens,
            temperature=sampling_temperature,
        )

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_gts,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        tok = tokenize_prompt_and_output(repeated_prompts, responses, tokenizer)
        input_ids = tok["input_ids"]
        labels = tok["labels"]
        response_mask = tok["response_mask"]

        old_log_probs = _compute_old_log_probs(
            policy, input_ids, labels, micro_train_batch_size, device
        )

        n_micro = rollout_batch_size // micro_train_batch_size
        epoch_losses: list[float] = []
        epoch_clip_fracs: list[float] = []
        last_grad_norm: float | None = None
        microbatch_counter = 0

        for _ in range(epochs_per_rollout_batch):
            order = list(range(rollout_batch_size))
            rng.shuffle(order)
            for mb_idx in range(n_micro):
                idx = order[mb_idx * micro_train_batch_size : (mb_idx + 1) * micro_train_batch_size]
                ids_mb = input_ids[idx].to(device)
                labels_mb = labels[idx].to(device)
                mask_mb = response_mask[idx].to(device)
                old_lp_mb = old_log_probs[idx].to(device)
                adv_mb = advantages[idx].unsqueeze(-1).to(device)

                policy_lp = get_response_log_probs(
                    policy, ids_mb, labels_mb, return_token_entropy=False
                )["log_probs"]

                loss, mb_meta = grpo_microbatch_train_step(
                    policy_log_probs=policy_lp,
                    response_mask=mask_mb,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=adv_mb,
                    old_log_probs=old_lp_mb,
                    cliprange=cliprange,
                )
                epoch_losses.append(float(loss.item()))
                epoch_clip_fracs.append(float(mb_meta["clip_fraction"].item()))
                microbatch_counter += 1

                if microbatch_counter % gradient_accumulation_steps == 0:
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        step_time = time.time() - t0
        step_metrics = {
            "step": step,
            "loss_mean": float(sum(epoch_losses) / len(epoch_losses)),
            "clip_fraction": float(sum(epoch_clip_fracs) / len(epoch_clip_fracs)),
            "grad_norm": last_grad_norm,
            "step_time_s": step_time,
            "reward_mean": reward_meta["reward_mean"],
            "format_reward_mean": reward_meta["format_reward_mean"],
            "answer_reward_mean": reward_meta["answer_reward_mean"],
            "group_std_mean": reward_meta["group_std_mean"],
        }
        metrics_log.append(step_metrics)
        with metrics_path.open("a") as f:
            f.write(json.dumps(step_metrics) + "\n")

        print(
            f"[step {step:3d}] loss={step_metrics['loss_mean']:.4f} "
            f"r={reward_meta['reward_mean']:.3f} "
            f"fmt={reward_meta['format_reward_mean']:.3f} "
            f"ans={reward_meta['answer_reward_mean']:.3f} "
            f"clip={step_metrics['clip_fraction']:.3f} "
            f"gn={last_grad_norm if last_grad_norm is None else f'{last_grad_norm:.2f}'} "
            f"({step_time:.1f}s)"
        )

        # Save the first 4 rollouts of every step for the writeup.
        with rollouts_path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "examples": [
                            {
                                "prompt": repeated_prompts[i],
                                "response": responses[i],
                                "reward": float(raw_rewards[i].item()),
                                "ground_truth": repeated_gts[i],
                            }
                            for i in range(min(4, len(responses)))
                        ],
                    }
                )
                + "\n"
            )

        if (step + 1) % eval_every == 0 or step == n_grpo_steps - 1:
            _load_policy_into_vllm(policy, llm)
            val_metrics = _evaluate(
                llm, val_examples, max_tokens=sampling_max_tokens, temperature=sampling_temperature
            )
            val_metrics["step"] = step
            val_log.append(val_metrics)
            with val_path.open("a") as f:
                f.write(json.dumps(val_metrics) + "\n")
            print(
                f"  [val @ {step}] ans_acc={val_metrics['answer_accuracy']:.4f} "
                f"fmt_acc={val_metrics['format_accuracy']:.4f}"
            )

    if save_final:
        save_dir = output_dir / "final_policy"
        policy.save_pretrained(save_directory=str(save_dir))
        tokenizer.save_pretrained(save_directory=str(save_dir))
        print(f"saved final policy -> {save_dir}")

    return {"metrics": metrics_log, "validation": val_log}


def _bool_arg(v: str) -> bool:
    return v.lower() in {"true", "1", "yes", "y"}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO training loop (§3.5).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--n-grpo-steps", type=int, default=50)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--advantage-eps", type=float, default=1e-6)
    p.add_argument("--rollout-batch-size", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--sampling-temperature", type=float, default=1.0)
    p.add_argument("--sampling-max-tokens", type=int, default=256)
    p.add_argument("--sampling-min-tokens", type=int, default=4)
    p.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--cliprange", type=float, default=1.0)
    p.add_argument("--normalize-by-std", type=_bool_arg, default=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-save-final", action="store_true")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--attn-impl", default="flash_attention_2")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    train_grpo(
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        sampling_min_tokens=args.sampling_min_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cliprange=args.cliprange,
        normalize_by_std=args.normalize_by_std,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        eval_size=args.eval_size,
        seed=args.seed,
        save_final=not args.no_save_final,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        attn_impl=args.attn_impl,
    )


if __name__ == "__main__":
    main()
