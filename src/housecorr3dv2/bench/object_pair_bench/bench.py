from __future__ import annotations

import sys
from pathlib import Path

import yaml

from housecorr3dv2.bench.bench import Bench, BenchConfig, register_bench


@register_bench("ObjectPairBench")
class ObjectPairBench(Bench):
    """Evaluation bench for object-pair correspondence benchmarks."""

    def run(self, dataset_cfg, method_cfg, task_cfg) -> dict:
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader
        from o3b.dataset.dataset import build_dataset
        from o3b.data.datatypes.object import collate_object_pairs
        from o3b.task.task import build_task
        from housecorr3dv2.method.method import build_method

        dataset = build_dataset(dataset_cfg)
        print(f"Dataset: {dataset_cfg.class_name}  ({len(dataset)} items)")

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_object_pairs,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        method = build_method(method_cfg)
        task = build_task(OmegaConf.create(task_cfg))
        print(f"Method:  {method_cfg.class_name}")
        print(f"Task:    {task_cfg['class_name']}")
        print(f"Eval:    batch_size={self.cfg.batch_size}  n_batches={len(loader)}\n")

        accum: dict[str, list] = {}
        n_samples = 0

        for batch_idx, batch in enumerate(loader):
            quant, _ = task(method(batch))

            B = (batch.src_obj_kpts3d.shape[0]
                 if batch.src_obj_kpts3d is not None else self.cfg.batch_size)
            n_samples += B

            for metric_name, value in quant.mean().items():
                accum.setdefault(metric_name, []).append(value)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"  [{batch_idx + 1:4d}/{len(loader)}]  samples={n_samples}", end="")
                for k, vals in accum.items():
                    print(f"  {k}={sum(vals)/len(vals):.4f}", end="")
                print()

        print(f"\n{'─'*50}")
        print(f"Results  ({n_samples} samples)")
        results: dict[str, float] = {}
        for k, vals in accum.items():
            mean_val = sum(vals) / len(vals)
            results[k] = mean_val
            print(f"  {k:<25} {mean_val:.4f}")

        return results

    @classmethod
    def eval_cli(cls, config_path: Path) -> None:
        from o3b.dataset.dataset import DatasetConfig
        from housecorr3dv2.method.method import MethodConfig

        config_path = Path(config_path)
        if config_path.suffix not in (".yaml", ".yml"):
            config_path = config_path.with_suffix(".yaml")
        if not config_path.exists():
            print(f"ERROR: config not found: {config_path}", file=sys.stderr)
            sys.exit(1)

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        dataset_cfg = DatasetConfig.from_dict(cfg["dataset"])
        method_cfg = MethodConfig.from_dict(cfg["method"])
        task_cfg = cfg["task"]

        bench_section = dict(cfg.get("bench", {}))
        bench_section.setdefault("class_name", "ObjectPairBench")
        bench_cfg = BenchConfig.from_dict(bench_section)

        bench = cls(bench_cfg)
        bench.run(dataset_cfg, method_cfg, task_cfg)

    @classmethod
    def main(cls, argv=None) -> None:
        import argparse
        parser = argparse.ArgumentParser(prog="hc3d-bench", description="ObjectPairBench eval")
        parser.add_argument("--config", required=True, type=Path, metavar="YAML",
                            help="Path to bench config YAML")
        args = parser.parse_args(argv)
        cls.eval_cli(args.config)
