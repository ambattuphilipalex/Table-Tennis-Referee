from __future__ import annotations

from housecorr3dv2.bench.bench import Bench, register_bench


@register_bench("FrameObjectPairBench")
class FrameObjectPairBench(Bench):
    """Evaluation bench for frame + object-pair benchmarks."""

    def run(self, dataset_cfg, method_cfg) -> dict:
        raise NotImplementedError("FrameObjectPairBench.run() is not yet implemented.")
