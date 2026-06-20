from __future__ import annotations

import importlib
from dataclasses import dataclass, field


@dataclass
class BenchConfig:
    class_name: str
    batch_size: int = 4
    num_workers: int = 0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchConfig":
        d = dict(d)
        class_name = d.pop("class_name")
        batch_size = d.pop("batch_size", 4)
        num_workers = d.pop("num_workers", 0)
        extra = d.pop("extra", {})
        extra.update(d)
        return cls(class_name=class_name, batch_size=batch_size, num_workers=num_workers, extra=extra)


# ── Bench registry ─────────────────────────────────────────────────────────────

_REGISTRY_BENCHES: dict[str, type["Bench"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "ObjectPairBench":      "housecorr3dv2.bench.object_pair_bench.bench",
    "FrameObjectPairBench": "housecorr3dv2.bench.frame_object_pair_bench.bench",
}


def _ensure_bench_imported(name: str) -> None:
    if name not in _REGISTRY_BENCHES and name in _CLASS_TO_MODULE:
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_bench(name: str):
    """Class decorator: @register_bench("ObjectPairBench")"""
    def decorator(cls):
        _REGISTRY_BENCHES[name] = cls
        return cls
    return decorator


def build_bench(cfg: BenchConfig) -> "Bench":
    _ensure_bench_imported(cfg.class_name)
    if cfg.class_name not in _REGISTRY_BENCHES:
        raise KeyError(
            f"Unknown bench '{cfg.class_name}'. "
            f"Registered: {sorted(_REGISTRY_BENCHES)}"
        )
    return _REGISTRY_BENCHES[cfg.class_name](cfg)


# ── Base class ─────────────────────────────────────────────────────────────────

class Bench:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg

    def run(self, dataset_cfg, method_cfg, task_cfg) -> dict:
        raise NotImplementedError

    @classmethod
    def eval_cli(cls, config_path) -> None:
        raise NotImplementedError(f"{cls.__name__} does not implement eval_cli()")
