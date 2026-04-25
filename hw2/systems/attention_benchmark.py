from __future__ import annotations

import argparse
import csv
import gc
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Same basics-path bootstrap as benchmark.py — make sure we get the regular
# `basics` package and not the outer namespace dir.
_BASICS_PARENT = Path(__file__).resolve().parent.parent / "basics"
if str(_BASICS_PARENT) not in sys.path:
    sys.path.insert(0, str(_BASICS_PARENT))
_cached = sys.modules.get("basics")
if _cached is not None and getattr(_cached, "__file__", None) is None:
    for _name in [n for n in list(sys.modules) if n == "basics" or n.startswith("basics.")]:
        del sys.modules[_name]

import torch  # noqa: E402

from basics.model import scaled_dot_product_attention  # noqa: E402


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    warmup_passes: int = 5
    compile_attention: bool = False
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_qkv(
    batch_size: int, sequence_length: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random Q, K, V with grad enabled. num_heads=1 (single head, per §2.7)."""
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(shape, device=device, requires_grad=True)
    k = torch.randn(shape, device=device, requires_grad=True)
    v = torch.randn(shape, device=device, requires_grad=True)
    return q, k, v


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_fn,
    forward_passes: int,
    backward_passes: int,
    warmup_passes: int,
) -> dict[str, float]:
    """Time forward and backward passes; capture memory in use right before backward."""
    # Warm up: a few full forward+backward iterations to stabilize allocator/kernels.
    for _ in range(warmup_passes):
        out = attention_fn(q, k, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    _sync()

    # Forward-only timing.
    fwd_start = timeit.default_timer()
    for _ in range(forward_passes):
        out = attention_fn(q, k, v)
        _sync()
    fwd_time = (timeit.default_timer() - fwd_start) / forward_passes

    # One forward pass kept around so we can measure memory before backward.
    out = attention_fn(q, k, v)
    _sync()
    if torch.cuda.is_available():
        mem_before_bwd_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        mem_before_bwd_mb = 0.0

    # Backward timing (each iteration: fresh forward to rebuild the graph).
    total_bwd = 0.0
    for i in range(backward_passes):
        if i > 0:
            out = attention_fn(q, k, v)
            _sync()
        bwd_start = timeit.default_timer()
        out.sum().backward()
        _sync()
        total_bwd += timeit.default_timer() - bwd_start
        q.grad = k.grad = v.grad = None
    bwd_time = total_bwd / backward_passes

    return {
        "forward_s": fwd_time,
        "backward_s": bwd_time,
        "mem_before_backward_mb": mem_before_bwd_mb,
    }


def benchmark_attention_grid(
    config: AttentionBenchmarkConfig,
) -> list[dict[str, float | int | str]]:
    """Run the §2.7 cartesian product. OOMs are recorded as a row with status=oom."""
    device = _device()
    attention_fn = scaled_dot_product_attention
    if config.compile_attention:
        attention_fn = torch.compile(attention_fn)

    rows: list[dict[str, float | int | str]] = []
    for head_dim, seq_len in iter_benchmark_shapes(config):
        q = k = v = None
        try:
            q, k, v = make_qkv(config.batch_size, seq_len, head_dim, device)
            stats = benchmark_attention_once(
                q, k, v, attention_fn,
                config.forward_passes, config.backward_passes, config.warmup_passes,
            )
            row = {
                "head_dim": head_dim,
                "sequence_length": seq_len,
                "batch_size": config.batch_size,
                "compile_attention": config.compile_attention,
                "status": "ok",
                **stats,
            }
            print(
                f"d={head_dim:3d} seq={seq_len:5d} -> "
                f"fwd={stats['forward_s'] * 1000:.3f} ms  "
                f"bwd={stats['backward_s'] * 1000:.3f} ms  "
                f"mem_before_bwd={stats['mem_before_backward_mb']:.1f} MB"
            )
        except torch.cuda.OutOfMemoryError:
            row = {
                "head_dim": head_dim,
                "sequence_length": seq_len,
                "batch_size": config.batch_size,
                "compile_attention": config.compile_attention,
                "status": "oom",
                "forward_s": float("nan"),
                "backward_s": float("nan"),
                "mem_before_backward_mb": float("nan"),
            }
            print(f"d={head_dim:3d} seq={seq_len:5d} -> OOM")
        finally:
            del q, k, v
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "compiled" if config.compile_attention else "vanilla"
    out_path = config.output_dir / f"attention_grid_{suffix}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved -> {out_path}")
    return rows


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(
        compile_attention=args.compile_attention,
        output_dir=args.output_dir,
    )
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
