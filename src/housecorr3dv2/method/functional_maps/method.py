from __future__ import annotations

from housecorr3dv2.method.method import Method, register_method


@register_method("FunctionalMapsMethod")
class FunctionalMapsMethod(Method):
    """Functional maps-based 3D shape correspondence method."""

    def __init__(self, num_eigenvectors: int = 50, **kwargs):
        self.num_eigenvectors = num_eigenvectors

    def forward(self, batch):
        raise NotImplementedError(
            "FunctionalMapsMethod.forward() is not yet implemented."
        )
