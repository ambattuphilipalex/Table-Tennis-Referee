# fetch_data.py — download OpenTT into ./data, bypassing the o3b CLI
from pathlib import Path
from o3b.dataset.dataset import DatasetConfig
from o3b.dataset.opentt import OpenTT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

cfg = DatasetConfig(
    class_name="OpenTT",
    path_raw=DATA / "OpenTT",
    path_preprocess=DATA / "OpenTT_Preprocess",
    split="all",
    scene_length=64,
)

OpenTT.fetch(cfg)   # videos + annotations -> data/OpenTT
OpenTT.index(cfg)   # manifest.json        -> data/OpenTT_Preprocess
print("done ->", DATA)