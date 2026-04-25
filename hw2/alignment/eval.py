from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .drgrpo_grader import r1_zero_reward_fn
from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets.

    Returns a list of dicts with keys ``question`` and ``answer`` (the raw
    chain-of-thought string ending in ``#### <number>``). The numeric ground
    truth is parsed out as ``ground_truth``.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[dict[str, Any]] = []
    for row in ds:
        answer_text = row["answer"]
        # GSM8K answers end with "#### <number>"; everything after is the gold.
        if "####" in answer_text:
            ground_truth = answer_text.split("####")[-1].strip().replace(",", "")
        else:
            ground_truth = answer_text.strip()
        examples.append(
            {
                "question": row["question"],
                "answer": answer_text,
                "ground_truth": ground_truth,
            }
        )
    return examples


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=ex["question"]) for ex in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable artifacts.

    ``ground_truths`` is required to score outputs; we accept it as an extra
    argument so this function stays close to the signature the PDF suggests.
    """
    if ground_truths is None:
        raise ValueError("evaluate_vllm needs ground_truths to score outputs")
    if len(prompts) != len(ground_truths):
        raise ValueError("prompts and ground_truths must have the same length")

    outputs = vllm_model.generate(list(prompts), eval_sampling_params)

    records: list[dict[str, Any]] = []
    bucket_counts = Counter()
    for prompt, gt, out in zip(prompts, ground_truths, outputs):
        generation = out.outputs[0].text
        reward = reward_fn(generation, gt)
        bucket = _bucket_for(reward)
        bucket_counts[bucket] += 1
        records.append(
            {
                "prompt": prompt,
                "ground_truth": gt,
                "generation": generation,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
                "bucket": bucket,
            }
        )

    n = len(records)
    summary = {
        "n": n,
        "accuracy": bucket_counts["correct"] / n if n else 0.0,
        "format_accuracy": (bucket_counts["correct"] + bucket_counts["format_only"]) / n if n else 0.0,
        "counts": {
            "correct": bucket_counts["correct"],
            "format_only": bucket_counts["format_only"],
            "no_format": bucket_counts["no_format"],
        },
    }
    return {"records": records, "summary": summary}


def _bucket_for(reward: dict[str, float]) -> str:
    """Map a reward dict to one of the three §3.1.2 categories."""
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
        return "correct"
    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0:
        return "format_only"
    return "no_format"


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis.

    Writes per-example records to ``output_path`` (JSONL) and a small summary
    to ``output_path`` with the ``.summary.json`` suffix.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in results["records"]:
            f.write(json.dumps(record) + "\n")

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w") as f:
        json.dump(results["summary"], f, indent=2)


def run_direct_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "test",
    max_examples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate the direct-prediction GSM8K baseline from §3.1.2."""
    from vllm import LLM, SamplingParams

    examples = load_gsm8k_examples(split)
    if max_examples is not None:
        examples = examples[:max_examples]
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    ground_truths = [ex["ground_truth"] for ex in examples]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=seed,
    )

    llm = LLM(model=model_name)
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params, ground_truths)
    write_evaluation_results(results, output_path)
    print(
        f"[direct] n={results['summary']['n']}  "
        f"acc={results['summary']['accuracy']:.4f}  "
        f"counts={results['summary']['counts']}"
    )
    print(f"saved -> {output_path}")
    return results


def run_cot_baseline(output_path: Path) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    raise NotImplementedError


def run_self_consistency_baseline(output_path: Path, k: int = 5) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    raise NotImplementedError


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSM8K direct-prediction baseline (§3.1.2).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts") / "sec3_1" / "direct_baseline.jsonl",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    run_direct_baseline(
        output_path=args.output_path,
        model_name=args.model_name,
        split=args.split,
        max_examples=args.max_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
