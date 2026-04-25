from __future__ import annotations

import argparse
import json
import statistics
import sys
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# The real `basics` package lives at <repo>/basics/basics/. Make sure the parent
# dir is on sys.path so `import basics.model` resolves to the regular package,
# not the outer namespace dir. Also evict a stale namespace import if one snuck
# in from an earlier `import basics`.
_BASICS_PARENT = Path(__file__).resolve().parent.parent / "basics"
if str(_BASICS_PARENT) not in sys.path:
    sys.path.insert(0, str(_BASICS_PARENT))
_cached = sys.modules.get("basics")
if _cached is not None and getattr(_cached, "__file__", None) is None:
    for _name in [n for n in list(sys.modules) if n == "basics" or n.startswith("basics.")]:
        del sys.modules[_name]

import math  # noqa: E402

import torch  # noqa: E402
import torch.cuda.nvtx as nvtx  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import einsum  # noqa: E402

import basics.model  # noqa: E402
from basics.model import BasicsTransformerLM  # noqa: E402
from basics.nn_utils import softmax  # noqa: E402
from basics.optimizer import AdamW  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    use_annotated_attention: bool = False
    compile_model: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--use-annotated-attention", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10000.0,
    )
    model = model.to(_device())
    if config.compile_model:
        model = torch.compile(model)
    return model


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        0, config.vocab_size, (config.batch_size, config.context_length), device=device
    )


def _compute_loss(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    # Standard next-token prediction loss on the random batch.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = batch[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
    )


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Execute one benchmark step and synchronize CUDA before returning."""
    if mode == "forward":
        with torch.no_grad(), autocast_context, nvtx.range("forward"):
            model(batch)
    elif mode == "forward-backward":
        with autocast_context, nvtx.range("forward"):
            logits = model(batch)
            loss = _compute_loss(logits, batch)
        with nvtx.range("backward"):
            loss.backward()
        model.zero_grad(set_to_none=True)
    elif mode == "train-step":
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        with autocast_context, nvtx.range("forward"):
            logits = model(batch)
            loss = _compute_loss(logits, batch)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("optimizer-step"):
            optimizer.step()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    _sync()


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    if config.use_annotated_attention:
        # Swap basics' attention for the NVTX-annotated version. Math is unchanged.
        basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    device = _device()
    model = build_model(config)
    batch = make_random_batch(config, device)
    autocast_ctx = make_autocast_context(config.use_bf16)

    optimizer = None
    if config.mode == "train-step":
        optimizer = AdamW(model.parameters(), lr=1e-4)

    # Warm-up: kick off kernels, let caching allocator stabilize, compile if needed.
    with nvtx.range("warmup"):
        for _ in range(config.warmup_steps):
            run_single_step(model, batch, config.mode, autocast_ctx, optimizer)

    maybe_start_memory_history(config.use_memory_profiler)

    timings: list[float] = []
    with nvtx.range("measure"):
        for _ in range(config.measure_steps):
            start = timeit.default_timer()
            run_single_step(model, batch, config.mode, autocast_ctx, optimizer)
            timings.append(timeit.default_timer() - start)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    maybe_dump_memory_snapshot(
        config.use_memory_profiler, config.output_dir / "memory_snapshot.pickle"
    )

    mean = statistics.fmean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    results = {
        "mean_s": mean,
        "stdev_s": stdev,
        "min_s": min(timings),
        "max_s": max(timings),
        "n": float(len(timings)),
    }
    print(
        f"[{config.model_size}/{config.mode}] "
        f"ctx={config.context_length} bs={config.batch_size} bf16={config.use_bf16} "
        f"mean={mean * 1000:.2f} ms  std={stdev * 1000:.2f} ms  "
        f"min={min(timings) * 1000:.2f} ms  max={max(timings) * 1000:.2f} ms  "
        f"(n={len(timings)})"
    )

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_size": config.model_size,
        "mode": config.mode,
        "context_length": config.context_length,
        "batch_size": config.batch_size,
        "vocab_size": config.vocab_size,
        "warmup_steps": config.warmup_steps,
        "measure_steps": config.measure_steps,
        "use_bf16": config.use_bf16,
        "compile_model": config.compile_model,
        "device": str(_device()),
        **results,
        "timings_s": timings,
    }
    log_path = config.output_dir / "benchmarks.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"saved -> {log_path}")
    return results


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Drop-in replacement for basics.model.scaled_dot_product_attention with NVTX ranges."""
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
    with nvtx.range("computing softmax"):
        weights = softmax(scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(weights, V, "... query key, ... key d_v -> ... query d_v")


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)


def make_autocast_context(use_bf16: bool):
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        use_annotated_attention=args.use_annotated_attention,
        compile_model=args.compile_model,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
