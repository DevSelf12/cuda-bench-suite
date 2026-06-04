"""CLI entry point for cuda-bench-suite."""

import argparse
import sys
import json
from pathlib import Path

from cubench import __version__
from cubench.config import load_config
from cubench.runner import BenchmarkRunner
from cubench.device import get_device_info


def main():
    parser = argparse.ArgumentParser(
        prog="cubench",
        description="CUDA Benchmark & Profiling Toolkit for Nvidia GPUs",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run benchmarks from config file")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.add_argument("--output", default="results/", help="Output directory")
    p_run.add_argument("--device", type=int, default=0, help="GPU device index")

    # ---- matmul ----
    p_mm = sub.add_parser("matmul", help="Run matrix multiplication benchmark")
    p_mm.add_argument("--precision", choices=["fp16", "bf16", "fp32", "tf32", "fp64"], default="fp32")
    p_mm.add_argument("--size", type=int, default=4096, help="Matrix size (NxN)")
    p_mm.add_argument("--iterations", type=int, default=100)
    p_mm.add_argument("--warmup", type=int, default=10)
    p_mm.add_argument("--device", type=int, default=0)

    # ---- bandwidth ----
    p_bw = sub.add_parser("bandwidth", help="Run memory bandwidth benchmark")
    p_bw.add_argument("--direction", choices=["h2d", "d2h", "d2d", "all"], default="all")
    p_bw.add_argument("--size", default="1gb", help="Transfer size (e.g. 256mb, 1gb)")
    p_bw.add_argument("--iterations", type=int, default=50)
    p_bw.add_argument("--device", type=int, default=0)

    # ---- tensorcore ----
    p_tc = sub.add_parser("tensorcore", help="Run Tensor Core benchmark")
    p_tc.add_argument("--op", choices=["bf16_gemm", "fp16_gemm", "tf32_gemm"], default="bf16_gemm")
    p_tc.add_argument("--m", type=int, default=4096)
    p_tc.add_argument("--n", type=int, default=4096)
    p_tc.add_argument("--k", type=int, default=4096)
    p_tc.add_argument("--iterations", type=int, default=100)
    p_tc.add_argument("--device", type=int, default=0)

    # ---- nccl ----
    p_nccl = sub.add_parser("nccl", help="Run NCCL multi-GPU benchmark")
    p_nccl.add_argument("--operation", choices=["allreduce", "allgather", "alltoall"], default="allreduce")
    p_nccl.add_argument("--sizes", default="1mb,64mb,1gb", help="Comma-separated message sizes")
    p_nccl.add_argument("--gpus", default=None, help="GPU indices (e.g. 0,1,2,3)")

    # ---- profile ----
    p_prof = sub.add_parser("profile", help="Profile a benchmark with Nsight")
    p_prof.add_argument("benchmark", help="Benchmark to profile")
    p_prof.add_argument("--size", type=int, default=4096)
    p_prof.add_argument("--ncu", action="store_true", help="Use Nsight Compute")
    p_prof.add_argument("--nsys", action="store_true", help="Use Nsight Systems")
    p_prof.add_argument("--output", default="profile.ncu-rep")

    # ---- info ----
    sub.add_parser("info", help="Show GPU device information")

    # ---- compare ----
    p_cmp = sub.add_parser("compare", help="Compare benchmark results")
    p_cmp.add_argument("files", nargs="+", help="Result JSON files to compare")
    p_cmp.add_argument("--format", choices=["text", "html", "csv"], default="text")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "info":
        _show_info()
    elif args.command == "run":
        _run_config(args)
    elif args.command == "compare":
        _compare(args)
    elif args.command in ("matmul", "bandwidth", "tensorcore", "nccl"):
        _run_single(args)
    elif args.command == "profile":
        _profile(args)


def _show_info():
    info = get_device_info()
    print(json.dumps(info, indent=2))


def _run_config(args):
    config = load_config(args.config)
    runner = BenchmarkRunner(device=args.device, output_dir=args.output)
    results = runner.run_all(config)
    print(f"\nResults saved to {args.output}/")


def _run_single(args):
    runner = BenchmarkRunner(device=getattr(args, "device", 0))
    name = args.command
    params = {k: v for k, v in vars(args).items() if k not in ("command", "device")}
    result = runner.run_single(name, **params)
    print(json.dumps(result, indent=2))


def _compare(args):
    from cubench.reporters.json_report import compare_results
    results = [json.loads(Path(f).read_text()) for f in args.files]
    compare_results(results, fmt=args.format)


def _profile(args):
    runner = BenchmarkRunner(device=0)
    runner.profile(args.benchmark, ncu=args.ncu, nsys=args.nsys, output=args.output, size=args.size)


if __name__ == "__main__":
    main()
