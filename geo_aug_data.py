"""Geometric + JPEG augmentation (model_4). Builds on data.py without modifying it.

Training images: random rescale -> small rotation -> JPEG re-save -> crop at any pixel offset -> flip.
Validation: fixed per-image rescale/rotation/off-grid crop, saved at q90 4:2:0 (cached in splits/).
"""
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import GRID, Config
from data import (VAL_JPEG_Q, VAL_JPEG_SUB, TrainValDataset, center_aligned_crop, make_split,
                  normalize, resave_jpeg, to_tensor)


@dataclass
class GeoConfig(Config):
    crop_align: int = 1                 # crop offset step: 1 = any pixel, 8 = old grid-aligned crops
    scale_p: float = 0.5                # probability of a random rescale
    scale_range: tuple = (0.75, 1.33)   # rescale factor range
    rot_p: float = 0.3                  # probability of a small random rotation
    rot_deg: float = 5.0                # max rotation angle in degrees


INTERPS = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)
VAL_GEO_SIZE = 384  # fixed output size so validation batches stack


def random_rescale(img, lo, hi, rng):
    s = math.exp(rng.uniform(math.log(lo), math.log(hi)))  # uniform in log-scale
    n = round(img.size[0] * s)
    return img.resize((n, n), rng.choice(INTERPS))

def rotate_inner(img, angle):
    """Rotate, then centre-crop the largest square that has no black corners."""
    w = img.size[0]
    r = img.rotate(angle, Image.BICUBIC)
    a = math.radians(abs(angle))
    s = int(w / (math.cos(a) + math.sin(a)))
    o = (w - s) // 2
    return r.crop((o, o, o + s, o + s))

def random_crop(img, size, rng, align):
    _, h, w = img.shape
    y = align * rng.randint(0, (h - size) // align)
    x = align * rng.randint(0, (w - size) // align)
    return img[:, y:y + size, x:x + size]


class GeoTrainValDataset(TrainValDataset):
    def __getitem__(self, i):
        cfg, rng = self.cfg, random  # global `random`; PyTorch seeds it differently in each worker
        pil = Image.open(self.img_dir / self.files[i]).convert("RGB")  # no 512 check: cached val is 384
        if self.train:
            if rng.random() < cfg.scale_p:
                pil = random_rescale(pil, *cfg.scale_range, rng)
            if rng.random() < cfg.rot_p:
                pil = rotate_inner(pil, rng.uniform(-cfg.rot_deg, cfg.rot_deg))
            if rng.random() < cfg.jpeg_p:
                pil = resave_jpeg(pil, quality=rng.randint(*cfg.jpeg_q),
                                  subsampling=rng.choice(cfg.jpeg_subsampling))
        img = to_tensor(pil)
        if self.train:
            img = random_crop(img, cfg.crop_size, rng, cfg.crop_align)
            if cfg.hflip and rng.random() < 0.5:
                img = img.flip(-1)
            if cfg.rot90:
                img = torch.rot90(img, rng.randint(0, 3), dims=(1, 2))
        elif cfg.val_mode == "center":
            img = center_aligned_crop(img, cfg.crop_size)
        return normalize(img), self.labels[i]


def val_geo_transform(img, rng):
    img = random_rescale(img, 0.9, 1.15, rng)
    img = img.rotate(rng.uniform(-3, 3), Image.BICUBIC)
    o = (img.size[0] - VAL_GEO_SIZE) // 2 + rng.randint(-7, 7)  # off-grid centre crop
    return img.crop((o, o, o + VAL_GEO_SIZE, o + VAL_GEO_SIZE))

def make_val_geo_cache(cfg, files) -> Path:
    out = cfg.split_dir / f"val_geo_q{VAL_JPEG_Q}_sub{VAL_JPEG_SUB}"
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        if not (out / f).exists():
            img = val_geo_transform(Image.open(cfg.train_dir / f).convert("RGB"), random.Random(10_000 + i))
            img.save(out / f, "JPEG", quality=VAL_JPEG_Q, subsampling=VAL_JPEG_SUB)
    return out


def build_geo_loaders(cfg: GeoConfig):
    """Returns train_dl (geometric + JPEG aug), val_dl (clean), val_geo_dl (geometric + q90 4:2:0)."""
    split = make_split(cfg)
    common = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(GeoTrainValDataset(cfg, split["train"], train=True), batch_size=cfg.batch_size,
                          shuffle=True, drop_last=True, generator=g, **common)
    val_dl = DataLoader(GeoTrainValDataset(cfg, split["val"], train=False),
                        batch_size=cfg.val_batch_size, shuffle=False, **common)
    geo_dir = make_val_geo_cache(cfg, split["val"])
    val_geo_dl = DataLoader(GeoTrainValDataset(cfg, split["val"], train=False, img_dir=geo_dir),
                            batch_size=cfg.val_batch_size, shuffle=False, **common)
    return train_dl, val_dl, val_geo_dl
