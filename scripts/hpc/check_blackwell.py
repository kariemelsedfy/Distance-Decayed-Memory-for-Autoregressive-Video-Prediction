#!/usr/bin/env python3
"""Probe bf16, SDPA backends, and FlexAttention on one CUDA device."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    return parser.parse_args()


def record_cuda_call(
    name: str, function: Callable[[], Any], torch: Any
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        output = function()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item())
        return {
            "name": name,
            "passed": finite,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "dtype": str(output.dtype),
            "shape": list(output.shape),
            "finite": finite,
        }
    except Exception as error:  # noqa: BLE001 - this is a compatibility probe
        return {
            "name": name,
            "passed": False,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    import torch
    import torch.nn.functional as functional
    from torch.nn.attention import SDPBackend, sdpa_kernel

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        report["fatal_error"] = "CUDA is not available"
        return report, False

    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    report["device"] = {
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    report["sdpa_flags"] = {
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
    }

    torch.manual_seed(0)
    shape = (1, 4, args.sequence_length, args.head_dim)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)

    checks: list[dict[str, Any]] = []
    checks.append(
        record_cuda_call("bf16_matmul", lambda: query @ key.transpose(-2, -1), torch)
    )

    backend_names = (
        "MATH",
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "CUDNN_ATTENTION",
    )
    for backend_name in backend_names:
        backend = getattr(SDPBackend, backend_name, None)
        if backend is None:
            checks.append(
                {
                    "name": f"sdpa_{backend_name.lower()}",
                    "passed": False,
                    "error_type": "UnavailableAPI",
                    "error": f"SDPBackend.{backend_name} is not defined",
                }
            )
            continue

        def run_sdpa(selected_backend: Any = backend) -> Any:
            with sdpa_kernel(selected_backend):
                return functional.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )

        checks.append(record_cuda_call(f"sdpa_{backend_name.lower()}", run_sdpa, torch))

    try:
        from torch.nn.attention.flex_attention import flex_attention

        compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
        checks.append(
            record_cuda_call(
                "flex_attention_compiled",
                lambda: compiled_flex_attention(query, key, value),
                torch,
            )
        )
    except Exception as error:  # noqa: BLE001 - import/compile is under test
        checks.append(
            {
                "name": "flex_attention_compiled",
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    report["checks"] = checks
    required = {
        "bf16_matmul",
        "sdpa_math",
        "sdpa_flash_attention",
        "sdpa_efficient_attention",
        "sdpa_cudnn_attention",
        "flex_attention_compiled",
    }
    passed_required = {check["name"] for check in checks if check.get("passed") is True}
    report["required_checks_passed"] = required <= passed_required
    return report, bool(report["required_checks_passed"])


def main() -> int:
    args = parse_args()
    report, passed = run_probe(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
