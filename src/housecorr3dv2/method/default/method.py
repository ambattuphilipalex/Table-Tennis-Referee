from __future__ import annotations

from housecorr3dv2.method.method import Method, register_method


@register_method("DefaultMethod")
class DefaultMethod(Method):
    """Identity method — returns the input batch unchanged."""

    def forward(self, batch):
        return batch
