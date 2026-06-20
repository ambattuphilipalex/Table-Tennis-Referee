"""Test Mesh.to_pytorch3d conversion and visualisation.

Loads one GLB from the pre-processed dataset, converts it to a PyTorch3D
Meshes object via the new ``Mesh.to_pytorch3d`` method, renders several views,
and saves PNGs to /tmp/test_mesh_to_pytorch3d/.

Run:
    cd /home/sommerl/PycharmProjects/housecorr3d
    python tests/test_mesh_to_pytorch3d.py
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O3B_SRC      = os.path.join(PROJECT_ROOT, "third_party", "o3b", "src")
sys.path.insert(0, O3B_SRC)

PATH_PREPROCESS = "/data/lmbraid19/sommerl/datasets/Omni6DPose_Preprocess"
MESH_TYPE       = "mc16_vuni4_r256_fdm"
OBJECT_ID       = "google_scan-backpack_0316"
OUT_DIR         = "/tmp/test_mesh_to_pytorch3d"


def test_mesh_to_pytorch3d() -> None:
    import torch
    import matplotlib.pyplot as plt
    from pathlib import Path
    from pytorch3d.structures.meshes import Meshes
    from pytorch3d.renderer.mesh.textures import Textures, TexturesUV
    from o3b.data.datatypes.mesh import Mesh
    from o3b.model.densematcher.densematcher.render import batch_render

    glb_path = Path(PATH_PREPROCESS) / "mesh" / MESH_TYPE / f"{OBJECT_ID}.glb"
    assert glb_path.exists(), f"Mesh not found: {glb_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load ──────────────────────────────────────────────────────────────────
    mesh = Mesh.load(glb_path)
    print(f"\nLoaded mesh:  verts={tuple(mesh.verts.shape)}  faces={tuple(mesh.faces.shape)}")
    print(f"  vert_colors : {mesh.vert_colors.shape if mesh.vert_colors is not None else None}")
    print(f"  texture     : {mesh.texture.shape if mesh.texture is not None else None}")
    print(f"  verts_uvs   : {mesh.verts_uvs.shape if mesh.verts_uvs is not None else None}")
    print(f"  faces_uvs   : {mesh.faces_uvs.shape if mesh.faces_uvs is not None else None}")

    # ── convert ───────────────────────────────────────────────────────────────
    pt3d_mesh = mesh.to_pytorch3d(device=device)

    assert isinstance(pt3d_mesh, Meshes), "to_pytorch3d must return a Meshes object"
    assert len(pt3d_mesh) == 1,           "batch size must be 1"
    assert pt3d_mesh.verts_list()[0].shape == mesh.verts.shape
    assert pt3d_mesh.faces_list()[0].shape == mesh.faces.shape
    assert pt3d_mesh.textures is not None, "textures must be set"

    tex_type = type(pt3d_mesh.textures).__name__
    print(f"\nPyTorch3D mesh: textures={tex_type}")

    # ── render ────────────────────────────────────────────────────────────────
    renders, _, _, _, _ = batch_render(device, pt3d_mesh, (4, 1), 512, 512,
                                       cameras=None, center=None)
    assert renders.shape[-1] == 4, "render output should be RGBA"
    print(f"Renders: {tuple(renders.shape)}  range=[{renders.min():.3f}, {renders.max():.3f}]")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(renders):
        img = frame[..., :3].clamp(0, 1).cpu().numpy()
        fpath = out_path / f"{OBJECT_ID}_view{i:02d}.png"
        plt.imsave(str(fpath), img)
        print(f"  saved {fpath}")

    print(f"\nPASS — {len(renders)} views saved to {OUT_DIR}")


def test_mesh_to_pytorch3d_normalised() -> None:
    """Same as above but on a normalised mesh (mirrors DenseMatcherModel usage)."""
    import torch
    import matplotlib.pyplot as plt
    from dataclasses import replace
    from pathlib import Path
    from o3b.data.datatypes.mesh import Mesh
    from o3b.model.densematcher.densematcher.render import batch_render

    glb_path = Path(PATH_PREPROCESS) / "mesh" / MESH_TYPE / f"{OBJECT_ID}.glb"
    assert glb_path.exists(), f"Mesh not found: {glb_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh  = Mesh.load(glb_path)
    verts = mesh.verts.float()
    center  = verts.mean(0)
    verts_c = verts - center
    scale   = verts_c.abs().max()
    if scale > 1e-8:
        verts_c = verts_c * (0.15 / scale)

    mesh_norm = replace(mesh, verts=verts_c)
    pt3d_mesh = mesh_norm.to_pytorch3d(device=device)

    v = pt3d_mesh.verts_list()[0]
    print(f"\nNormalised verts range: [{v.min():.4f}, {v.max():.4f}]  (expected ≈ ±0.15)")
    assert v.abs().max().item() <= 0.16, "normalised verts should be within ±0.15"

    renders, _, _, _, _ = batch_render(device, pt3d_mesh, (4, 1), 512, 512,
                                       cameras=None, center=None)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(renders):
        img = frame[..., :3].clamp(0, 1).cpu().numpy()
        fpath = out_path / f"{OBJECT_ID}_norm_view{i:02d}.png"
        plt.imsave(str(fpath), img)
        print(f"  saved {fpath}")

    print(f"\nPASS — normalised mesh {len(renders)} views saved to {OUT_DIR}")


if __name__ == "__main__":
    test_mesh_to_pytorch3d()
    test_mesh_to_pytorch3d_normalised()
