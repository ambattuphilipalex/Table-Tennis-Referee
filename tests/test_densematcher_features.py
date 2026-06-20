"""Test DenseMatcher per-vertex features via the o3b config pipeline.

Loads one object from the HouseCorr3D dataset (hc3d_object.yaml) and runs
DenseMatcher (dm.yaml) in use_mv_features=True mode to expose both outputs:
  a) mv_features   – 768-dim raw SDDINO multiview features  (must be non-zero)
  b) out_norm      – 512-dim DiffusionNet-refined features   (must be non-zero)

Run:
    cd /home/sommerl/PycharmProjects/housecorr3d
    python tests/test_densematcher_features.py
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O3B_SRC      = os.path.join(PROJECT_ROOT, "third_party", "o3b", "src")
CONFIGS_DIR  = os.path.join(PROJECT_ROOT, "third_party", "o3b", "src", "configs")
sys.path.insert(0, O3B_SRC)

DATASET_CFG = os.path.join(CONFIGS_DIR, "dataset", "hc3d_object.yaml")
MODEL_CFG   = os.path.join(CONFIGS_DIR, "model", "dm.yaml")


# ── helpers ───────────────────────────────────────────────────────────────────

def _check(name: str, t: "torch.Tensor") -> bool:
    """Print one-line summary; return True if finite and non-trivially non-zero."""
    import torch
    t = t.float()
    nan_frac  = t.isnan().float().mean().item()
    inf_frac  = t.isinf().float().mean().item()
    zero_frac = (t.abs() < 1e-9).all(dim=-1).float().mean().item() if t.ndim >= 2 else float("nan")
    norm_mean = t.norm(dim=-1).mean().item() if t.ndim >= 2 else t.norm().item()
    ok = nan_frac == 0.0 and inf_frac == 0.0 and (t.ndim < 2 or zero_frac < 0.5)
    flag = "OK " if ok else "BAD"
    print(
        f"  [{flag}] {name:<22s}  shape={str(tuple(t.shape)):<18s}"
        f"  nan={nan_frac:.3f}  inf={inf_frac:.3f}"
        f"  zero_rows={zero_frac:.3f}  ||·||={norm_mean:.4f}"
        f"  range=[{t.min().item():.4f}, {t.max().item():.4f}]"
    )
    return ok


# ── main test ─────────────────────────────────────────────────────────────────

def test_densematcher_features() -> None:
    import torch
    from pathlib import Path
    from omegaconf import OmegaConf

    from o3b.dataset.dataset import DatasetConfig, build_dataset
    from o3b.model.model import OD3D_Model
    from o3b.data.datatypes.object import collate_objects

    print(f"\n{'='*70}")
    print("[TEST] DenseMatcher features — loaded via config")
    print(f"  dataset : {DATASET_CFG}")
    print(f"  model   : {MODEL_CFG}")
    print(f"{'='*70}")

    # ── dataset ───────────────────────────────────────────────────────────────
    print("\n[1] Loading dataset …")
    dataset_cfg = DatasetConfig.from_yaml(
        Path(DATASET_CFG),
        overrides=["filter_count_max=1"],   # only one object needed
    )
    dataset = build_dataset(dataset_cfg)
    assert len(dataset) > 0, "Dataset is empty — run 'o3b dataset index -d hc3d_object' first"
    obj = dataset[0]
    assert obj.mesh is not None, "Object has no mesh — check object_modalities in hc3d_object.yaml"
    print(f"  object_id : {obj.object_id}")
    print(f"  verts     : {tuple(obj.mesh.verts.shape)}")
    print(f"  faces     : {tuple(obj.mesh.faces.shape)}")

    batch = collate_objects([obj])
    batch.mesh = obj.mesh   # collate_objects stacks tensors only; Mesh must be set manually

    # ── model — run with use_mv_features=True to get both outputs ─────────────
    print("\n[2] Loading DenseMatcher model …")
    model_raw = OmegaConf.load(MODEL_CFG)
    # Override: expose raw SDDINO mv_features through the public output path.
    # We will manually inspect both mv_features and out_norm via return_mvfeatures.
    model = OD3D_Model.create_from_config(OmegaConf.merge(model_raw, {"use_mv_features": True}))
    model.eval()

    # ── forward ───────────────────────────────────────────────────────────────
    print("\n[3] Running forward pass …")
    with torch.no_grad():
        result = model(batch)

    # use_mv_features=True → verts3d_feats contains mv_features (768-dim)
    mv_feats = result.verts3d_feats   # (1, V, 768)
    assert mv_feats is not None, "model returned no verts3d_feats"
    print("\n[a] mv_features (SDDINO 768-dim):")
    mv_ok = _check("mv_features", mv_feats[0])

    # Re-run with use_mv_features=False to get DiffusionNet output
    print("\n[4] Re-running with use_mv_features=False …")
    model_dm = OD3D_Model.create_from_config(OmegaConf.merge(model_raw, {"use_mv_features": False}))
    model_dm._mesh_featurizer = model._mesh_featurizer   # reuse loaded weights
    model_dm.eval()
    with torch.no_grad():
        result_dm = model_dm(batch)

    out_feats = result_dm.verts3d_feats   # (1, V, 512)
    assert out_feats is not None, "model returned no verts3d_feats for DiffusionNet output"
    print("\n[b] out_norm (DiffusionNet 512-dim):")
    dm_ok = _check("out_norm", out_feats[0])

    # ── assertions ────────────────────────────────────────────────────────────
    print()
    assert mv_ok, "mv_features are degenerate (NaN / Inf / all-zero)"
    assert dm_ok, "out_norm features are degenerate (NaN / Inf / all-zero)"
    print("PASS — both mv_features and out_norm are finite and non-zero")


if __name__ == "__main__":
    test_densematcher_features()
