from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field


@dataclass
class MethodConfig:
    class_name: str
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "MethodConfig":
        d = dict(d)
        class_name = d.pop("class_name")
        extra = d.pop("extra", {})
        extra.update(d)
        return cls(class_name=class_name, extra=extra)


# ── Method registry ────────────────────────────────────────────────────────────

_REGISTRY_METHODS: dict[str, type["Method"]] = {}

_CLASS_TO_MODULE: dict[str, str] = {
    "DefaultMethod":        "housecorr3dv2.method.default.method",
    "FunctionalMapsMethod": "housecorr3dv2.method.functional_maps.method",
}


def _ensure_method_imported(name: str) -> None:
    if name not in _REGISTRY_METHODS and name in _CLASS_TO_MODULE:
        importlib.import_module(_CLASS_TO_MODULE[name])


def register_method(name: str):
    """Class decorator: @register_method("DefaultMethod")"""
    def decorator(cls):
        _REGISTRY_METHODS[name] = cls
        return cls
    return decorator


def build_method(cfg: MethodConfig) -> "Method":
    _ensure_method_imported(cfg.class_name)
    if cfg.class_name not in _REGISTRY_METHODS:
        raise KeyError(
            f"Unknown method '{cfg.class_name}'. "
            f"Registered: {sorted(_REGISTRY_METHODS)}"
        )
    return _REGISTRY_METHODS[cfg.class_name].create_from_config(cfg)


# ── Base class ─────────────────────────────────────────────────────────────────

class Method:
    def forward(self, batch):
        raise NotImplementedError

    def __call__(self, batch):
        return self.forward(batch)

    @classmethod
    def create_from_config(cls, cfg: MethodConfig) -> "Method":
        name = cfg.class_name
        _ensure_method_imported(name)
        if name not in _REGISTRY_METHODS:
            raise KeyError(
                f"Unknown method '{name}'. Registered: {sorted(_REGISTRY_METHODS)}"
            )
        method_cls = _REGISTRY_METHODS[name]
        spec = inspect.getfullargspec(method_cls.__init__)
        if spec.varkw is not None:
            return method_cls(**cfg.extra)
        keys = spec.args[1:]
        return method_cls(**{k: cfg.extra[k] for k in keys if k in cfg.extra})
