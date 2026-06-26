from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import DataLoader

from o3b.dataset.dataset import DatasetConfig, build_dataset
from o3b.data.datatypes import collate_scenes

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OPENTT_RAW = DATA_ROOT / "OpenTT"
OPENTT_PREPROCESS = DATA_ROOT / "OpenTT_Preprocess"


@dataclass
class ClipConfig:
    split: str = "train"
    scene_length: int = 16
    frame_stride: int = 4
    clip_stride: int | None = None


def build_opentt(cfg: ClipConfig):
    extra = {"frame_stride": cfg.frame_stride}
    if cfg.clip_stride is not None:
        extra["clip_stride"] = cfg.clip_stride

    ds_cfg = DatasetConfig(
        class_name="OpenTT",
        path_raw=OPENTT_RAW,
        path_preprocess=OPENTT_PREPROCESS,
        split=cfg.split,
        scene_length=cfg.scene_length,
        extra=extra,
    )
    return build_dataset(ds_cfg)


def opentt_loader(cfg: ClipConfig, batch_size: int = 4,
                  shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        build_opentt(cfg),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_scenes,
    )


if __name__ == "__main__":
    loader = opentt_loader(ClipConfig(split="train"), batch_size=4,
                           shuffle=False, num_workers=0)
    print(f"#clips in split: {len(loader.dataset)}") # pyright: ignore[reportArgumentType]

    batch = next(iter(loader))
    mask = batch.rgbs_mask

    print("rgbs      :", batch.rgbs.shape)
    print("events[0] :", batch.events[0])
    print("score[0,0]:", batch.scoreboards[0][0])