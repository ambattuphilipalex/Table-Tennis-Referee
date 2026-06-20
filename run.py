from __future__ import annotations

import argparse
from pathlib import Path


def run_cfg(
    benchmark: str = "hc3d_crsp3d_object_pair_nn",
    ablation: str | None = None,
    platform: str | None = None,
) -> None:
    """Run a benchmark config, optionally merged with an ablation YAML overlay.

    Args:
        benchmark: Config name resolved against src/configs/eval/, or a path to a YAML file.
        ablation:  Path to a single ablation YAML (relative to repo root), or None for baseline.
        platform:  Platform name override; None inherits from the benchmark's defaults.
    """
    from o3b.cli import _run_bench_run, _resolve_bench_config

    bench_path = _resolve_bench_config(benchmark)

    ablation_path: Path | None = None
    if ablation is not None:
        p = Path(ablation)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Ablation config not found: {p}")
        ablation_path = p

    args = argparse.Namespace(
        benchmark=bench_path,
        ablation=ablation_path,
        platform=platform,
        bench_command="run",
    )
    _run_bench_run(args)


if __name__ == "__main__":
    _BENCHMARK = "hc3d_crsp3d_object_pair_nn"

    # Mesh resolution ablations (mc16 is the base; mc32/mc64 are larger remeshes)
    for _mesh_type in (
        "mc16_vuni100_r256_fdiff3f",
        "mc32_vuni100_r256_fdiff3f",
        "mc64_vuni100_r256_fdiff3f",
    ):
        run_cfg(
            benchmark=_BENCHMARK,
            ablation=f"src/configs/ablation/mesh_type/{_mesh_type}.yaml",
        )

    # Feature-model / n_views ablations (mc16 fixed, vary views and backbone)
    for _mesh_type in (
        "mc16_vuni4_r256_fdiff3f",
        "mc16_vuni4_r256_fdinov2s",
    ):
        run_cfg(
            benchmark=_BENCHMARK,
            ablation=f"src/configs/ablation/feature_model/{_mesh_type}.yaml",
        )
