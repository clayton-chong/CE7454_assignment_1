"""Dataset, train/val split and grid-aligned transforms.

Run `python data.py` once to create the split file and sanity-check the pipeline.
"""
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import GRID, IMG_SIZE, NUM_CLASSES, Config

# Get the label from the photo name, e.g. '10004_1.jpg' -> 1
def parse_label(fname: str) -> int:
    return int(Path(fname).stem.split("_")[1])

# Class-stratified train/val split, saved to disk so every run uses the same one.
def make_split(cfg: Config) -> dict:
    if cfg.split_file.exists():
        return json.loads(cfg.split_file.read_text())

    files = sorted(p.name for p in cfg.train_dir.glob("*.jpg"))
    by_class = {c: [f for f in files if parse_label(f) == c] for c in range(NUM_CLASSES)}

    rng = random.Random(cfg.split_seed)
    split = {"train": [], "val": []}
    for c in range(NUM_CLASSES):
        fs = by_class[c][:]
        rng.shuffle(fs)
        split["val"] += sorted(fs[: cfg.val_per_class])
        split["train"] += sorted(fs[cfg.val_per_class:])

    cfg.split_dir.mkdir(parents=True, exist_ok=True)
    cfg.split_file.write_text(json.dumps(split, indent=1))
    return split

# Image Conversions - Used for testing
def load_pil(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    assert img.size == (IMG_SIZE, IMG_SIZE), f"unexpected size {img.size} for {path}"
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # (3, H, W) uint8

def load_jpeg(path: Path) -> torch.Tensor:
    return to_tensor(load_pil(path))

# Re-encode as JPEG in memory (subsampling: 0 = 4:4:4, 2 = 4:2:0)
def resave_jpeg(img: Image.Image, quality: int, subsampling: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# Transforms - All transforms operate on uint8 tensors of shape (3, H, W).

# Return a random offset in [0, max_off] that is a multiple of GRID
def aligned_offset(max_off: int, rng: random.Random) -> int:
    return GRID * rng.randint(0, max_off // GRID)

# Random aligned cropping
def random_aligned_crop(img: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    _, h, w = img.shape
    y, x = aligned_offset(h - size, rng), aligned_offset(w - size, rng)
    return img[:, y:y + size, x:x + size]

# Center aligned cropping
def center_aligned_crop(img: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = img.shape
    y = (h - size) // 2 // GRID * GRID
    x = (w - size) // 2 // GRID * GRID
    return img[:, y:y + size, x:x + size]


def normalize(img: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1]"""
    return img.float().div_(127.5).sub_(1.0)


# Datasets

# Class for training dataset
class TrainValDataset(Dataset):
    def __init__(self, cfg: Config, files: list, train: bool, img_dir: Path = None):
        self.cfg, self.files, self.train = cfg, files, train
        self.img_dir = img_dir or cfg.train_dir
        self.labels = [parse_label(f) for f in files]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        cfg = self.cfg
        pil = load_pil(self.img_dir / self.files[i])
        if self.train and random.random() < cfg.jpeg_p:
            pil = resave_jpeg(pil, quality=random.randint(*cfg.jpeg_q),
                            subsampling=random.choice(cfg.jpeg_subsampling))
        img = to_tensor(pil)
        if self.train:
            rng = random  # global `random`; PyTorch seeds it differently in each worker
            img = random_aligned_crop(img, cfg.crop_size, rng)
            if cfg.hflip and rng.random() < 0.5:
                img = img.flip(-1)
            if cfg.rot90:
                img = torch.rot90(img, rng.randint(0, 3), dims=(1, 2))
        elif cfg.val_mode == "center":
            img = center_aligned_crop(img, cfg.crop_size)
        return normalize(img), self.labels[i]

# Class for dev dataset
class DevDataset(Dataset):
    """Unlabelled images (dev / test). Returns (image, image_id). Always full 512 image;
    TTA is handled in predict.py."""

    def __init__(self, img_dir: Path):
        self.files = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return normalize(load_jpeg(self.files[i])), int(self.files[i].stem)

# Loaders

def build_loaders(cfg: Config):
    split = make_split(cfg)
    train_ds = TrainValDataset(cfg, split["train"], train=True)
    val_ds = TrainValDataset(cfg, split["val"], train=False)

    common = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                          generator=g, **common)
    val_dl = DataLoader(val_ds, batch_size=cfg.val_batch_size, shuffle=False, **common)
    return train_dl, val_dl

# Cache the validation set
VAL_JPEG_Q, VAL_JPEG_SUB = 90, 2  # matches the test set: quality 90, 4:2:0

def make_val_jpeg_cache(cfg: Config, files: list) -> Path:
    out = cfg.split_dir / f"val_q{VAL_JPEG_Q}_sub{VAL_JPEG_SUB}"
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        if not (out / f).exists():
            load_pil(cfg.train_dir / f).save(out / f, "JPEG", quality=VAL_JPEG_Q, subsampling=VAL_JPEG_SUB)
    return out

def build_val_jpeg_loader(cfg: Config) -> DataLoader:
    split = make_split(cfg)
    img_dir = make_val_jpeg_cache(cfg, split["val"])
    ds = TrainValDataset(cfg, split["val"], train=False, img_dir=img_dir)
    return DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False,
                      num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
