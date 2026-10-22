"""Experiment configuration.

Defaults live in `Config`. For a new experiment, override only what changes:

    from config import Config
    cfg = Config(exp_name="crop128", crop_size=128)

`cfg.to_dict()` gives a plain dict to save alongside each run.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NUM_CLASSES = 3
CLASS_NAMES = ("real", "model_a", "model_b")
IMG_SIZE = 512
GRID = 8  # JPEG block size

@dataclass
class Config:
    # Experiment
    exp_name: str = "baseline"
    seed: int = 42

    # Paths
    train_dir: Path = ROOT / "train"
    dev_dir: Path = ROOT / "dev"
    split_dir: Path = ROOT / "splits"
    run_dir: Path = ROOT / "runs"
    pred_dir: Path = ROOT / "predictions"

    # Split
    val_per_class: int = 500
    split_seed: int = 0  # kept separate from `seed` so the split never changes between runs

    # Augmentation
    crop_size: int = 256 # training crop; must be a multiple of GRID
    hflip: bool = True
    rot90: bool = False # random 0/90/180/270 rotation
    val_mode: str = "full" # "full" (whole 512 image) or "center" (center crop_size)
    jpeg_p: float = 0.0                 # probability of re-saving a training image as JPEG (0 = off)
    jpeg_q: tuple = (80, 95)            # random JPEG quality range (inclusive)
    jpeg_subsampling: tuple = (0, 2)    # chroma subsampling choices: 0 = 4:4:4, 2 = 4:2:0
    
    # Loader
    batch_size: int = 64
    val_batch_size: int = 32
    num_workers: int = 6

    # Optimisation
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3 # Used for LR scheduler
    label_smoothing: float = 0.1

    def __post_init__(self):
        assert self.crop_size % GRID == 0 and self.crop_size <= IMG_SIZE, "crop_size must be a multiple of 8, <= 512"
        assert self.val_mode in ("full", "center")

    @property
    def split_file(self) -> Path:
        return self.split_dir / f"split_seed{self.split_seed}_val{self.val_per_class}.json"

    @property
    def out_dir(self) -> Path:
        return self.run_dir / self.exp_name

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
