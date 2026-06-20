from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="hc3d", description="HouseCorr3D CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("bench", help="Run a benchmark (e.g. hc3d bench --config src/configs/bench/hc3d_crsp3d_object_pair_nn.yaml)")
    p_bench.add_argument(
        "--config", required=True, type=Path, metavar="YAML",
        help="Path to bench config YAML",
    )

    args = parser.parse_args(argv)

    if args.command == "bench":
        from housecorr3dv2.bench.object_pair_bench.bench import ObjectPairBench
        ObjectPairBench.eval_cli(args.config)
