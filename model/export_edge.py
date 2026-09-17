"""
model/export_edge.py
--------------------
Exports the trained Attention U-Net-Lite model to optimized TorchScript (.pt)
for zero-dependency edge inference in resource-constrained rural clinics.
Validates numerical parity against the PyTorch eager baseline and benchmarks
CPU inference latency, throughput, and memory footprint.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from model.unet_attention import AttentionUNetLite


class EdgeInferenceWrapper(nn.Module):
    """
    Inference-optimized wrapper module for edge deployment.
    Guarantees deterministic forward pass without attention map output overhead.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calls forward with return_attention_maps=False
        return self.model(x, return_attention_maps=False)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu"
) -> AttentionUNetLite:
    """
    Loads AttentionUNetLite from a .pt checkpoint file.
    Supports either full training checkpoint dictionaries (with 'model_state_dict')
    or raw state dictionaries.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = AttentionUNetLite(in_channels=1, num_classes=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint structure in {checkpoint_path}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 128, 128),
    device: str = "cpu"
) -> Tuple[torch.jit.ScriptModule, Dict[str, Any]]:
    """
    Traces and compiles the PyTorch model to TorchScript, validates numerical parity,
    and serializes the compiled artifact to disk.

    Returns:
        (traced_model, verification_metrics)
    """
    model.eval()
    wrapper = EdgeInferenceWrapper(model).to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    # Validate eager model forward pass
    with torch.no_grad():
        eager_output = wrapper(dummy_input)

    # Trace model
    traced_model = torch.jit.trace(wrapper, dummy_input)

    # Verify traced output matches eager output
    with torch.no_grad():
        traced_output = traced_model(dummy_input)

    max_diff = (eager_output - traced_output).abs().max().item()
    is_close = torch.allclose(eager_output, traced_output, atol=1e-5, rtol=1e-4)

    if not is_close:
        raise RuntimeError(
            f"TorchScript numerical verification failed! Max absolute difference: {max_diff:.2e}"
        )

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    traced_model.save(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Re-load from disk to ensure serialized artifact is intact and self-contained
    reloaded_model = torch.jit.load(output_path, map_location=device)
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_output = reloaded_model(dummy_input)
    reloaded_close = torch.allclose(eager_output, reloaded_output, atol=1e-5, rtol=1e-4)

    verification_metrics = {
        "output_path": output_path,
        "input_shape": list(input_shape),
        "output_shape": list(eager_output.shape),
        "numerical_parity": is_close and reloaded_close,
        "max_abs_difference": float(max_diff),
        "model_file_size_mb": round(file_size_mb, 2),
        "parameters_count": model.count_parameters() if hasattr(model, "count_parameters") else 0
    }

    return traced_model, verification_metrics


def benchmark_edge_latency(
    model_or_script: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 128, 128),
    num_warmup: int = 10,
    num_iterations: int = 50,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Measures CPU execution latency, throughput, and jitter for edge clinical inference.
    """
    model_or_script.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_or_script(dummy_input)

    # Timed benchmark iterations
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model_or_script(dummy_input)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

    latencies_np = np.array(latencies_ms)
    mean_ms = float(np.mean(latencies_np))
    std_ms = float(np.std(latencies_np))
    p50_ms = float(np.percentile(latencies_np, 50))
    p95_ms = float(np.percentile(latencies_np, 95))
    p99_ms = float(np.percentile(latencies_np, 99))
    throughput_fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "num_iterations": num_iterations,
        "device": device,
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "p99_latency_ms": round(p99_ms, 2),
        "throughput_slices_per_sec": round(throughput_fps, 1),
        "clinical_realtime_capable": mean_ms < 200.0  # <200ms per slice is real-time interactive
    }


def main():
    parser = argparse.ArgumentParser(description="Export AttentionUNetLite to TorchScript for Edge Telemedicine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/centralized_best.pt",
                        help="Path to trained PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="checkpoints/edge_model_traced.pt",
                        help="Output path for compiled TorchScript model")
    parser.add_argument("--results", type=str, default="results/edge_export_benchmark.json",
                        help="Path to save edge export benchmark JSON")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of latency benchmark iterations")
    args = parser.parse_args()

    print("=" * 65)
    print("PHASE 8: EDGE MODEL EXPORT & LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device="cpu")
    print(f"[+] Loaded AttentionUNetLite with {model.count_parameters():,} parameters")

    print(f"Exporting to TorchScript: {args.output}")
    traced_model, export_meta = export_to_torchscript(model, args.output, device="cpu")
    print(f"[+] Export successful!")
    print(f"    - Numerical Parity: {export_meta['numerical_parity']}")
    print(f"    - Max Absolute Difference: {export_meta['max_abs_difference']:.2e}")
    print(f"    - Serialized Size: {export_meta['model_file_size_mb']} MB")

    print(f"\nBenchmarking CPU Edge Latency ({args.iterations} iterations)...")
    latency_meta = benchmark_edge_latency(traced_model, num_iterations=args.iterations, device="cpu")
    print(f"[+] Latency Results:")
    print(f"    - Mean Latency: {latency_meta['mean_latency_ms']} ms/slice (+/- {latency_meta['std_latency_ms']} ms)")
    print(f"    - Median (P50): {latency_meta['p50_latency_ms']} ms")
    print(f"    - 95th Percentile: {latency_meta['p95_latency_ms']} ms")
    print(f"    - Throughput: {latency_meta['throughput_slices_per_sec']} slices/sec")
    print(f"    - Interactive Real-time: {'YES (Optimal)' if latency_meta['clinical_realtime_capable'] else 'NO'}")

    benchmark_summary = {
        "export_metadata": export_meta,
        "latency_benchmark": latency_meta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\n[+] Benchmark summary saved to: {args.results}")
    print("=" * 65)


if __name__ == "__main__":
    main()
