import dataclasses
import random
import numpy as np
import torch
import math
import torch.nn.functional as F

from config import CLASS_NAMES, NUM_CLASSES, Config
from data import build_loaders
from model import build_model, count_params

# Set everything to a fixed seed
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# Get cuda or mps device if available, otherwise run on CPU
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # plain fp32: channels_last / bf16 are slower on MPS
    return torch.device("cpu")

# Builds the AdamW optimizer
def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)

# LR scheduler - linear warmup then cosine decay
def build_scheduler(opt, cfg, steps_per_epoch):
    """Per-step schedule: linear warm-up, then cosine decay to 0."""
    warmup, total = cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# Evaluation steo function - outputs the loss and accuracy (overall & per-class)
@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Returns loss, accuracy, per-class accuracy and confusion matrix (rows = true, cols = pred)."""
    model.eval()
    tot_loss, n = 0.0, 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        tot_loss += F.cross_entropy(logits, y, reduction="sum").item()
        idx = (y * NUM_CLASSES + logits.argmax(1)).cpu()
        cm += torch.bincount(idx, minlength=NUM_CLASSES**2).view(NUM_CLASSES, NUM_CLASSES)
        n += x.shape[0]
    per_class = (cm.diag().float() / cm.sum(1).clamp(min=1)).tolist()
    return {"loss": tot_loss / n, "acc": cm.diag().sum().item() / n, "per_class": per_class, "cm": cm}

# Builds confusion matrix
def format_cm(cm):
    w = max(len(c) for c in CLASS_NAMES)
    lines = [" " * (w + 8) + " ".join(f"{c:>{w}s}" for c in CLASS_NAMES)]
    for i, c in enumerate(CLASS_NAMES):
        lines.append(f"  true {c:>{w}s} " + " ".join(f"{v:>{w}d}" for v in cm[i].tolist()))
    return "\n".join(lines)

# Saves model checkpoint
def save_checkpoint(path, model, opt, sched, epoch, best_acc, cfg):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_acc": best_acc, "cfg": cfg.to_dict()}, path)

# Loads model checkpoint
def load_model(ckpt_path, device="cpu"):
    """Rebuild a model (and its Config) from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # keep only fields Config still has (older checkpoints may contain removed ones, e.g. highpass)
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ckpt["cfg"].items()
                    if k in fields and k not in ("train_dir", "dev_dir", "split_dir", "run_dir")})
    model = build_model().to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg