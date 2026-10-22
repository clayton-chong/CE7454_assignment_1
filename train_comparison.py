"""Ablation study: retrain model_4 four times, each with ONE augmentation removed.

    ablate_crop_align8 : crops at multiples of 8 px again (crop_align=8)
    ablate_no_jpeg     : no JPEG re-save                 (jpeg_p=0)
    ablate_no_rescale  : no random rescale               (scale_p=0)
    ablate_no_rotation : no small rotations              (rot_p=0)

Everything else (model, optimiser, schedule, 50 epochs, seed, split, geometric val set) is identical
to runs/model_4, so each run differs from model_4 in exactly one setting. Same loop as train.ipynb.
Outputs: comparison/<name>/{config.json, log.csv, best.pt, last.pt} and comparison/summary.csv.

    caffeinate -i python train_comparison.py             # all four (~6.5 h), keeps the Mac awake
    python train_comparison.py --only ablate_no_jpeg     # a single run
    python train_comparison.py --smoke                   # pipeline check, a few batches per epoch

Interrupted runs resume from last.pt when the script is started again; finished runs are skipped.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import CLASS_NAMES, ROOT
from geo_aug_data import GeoConfig, build_geo_loaders
from helpers import (build_optimizer, build_scheduler, evaluate, get_device, save_checkpoint,
                     seed_everything)
from model import build_model

# model_4 settings (runs/model_4/config.json); each ablation overrides one field
BASE = dict(crop_size=256, epochs=50, jpeg_p=0.7, jpeg_q=(80, 95), jpeg_subsampling=(0, 2),
            crop_align=1, scale_p=0.5, scale_range=(0.75, 1.33), rot_p=0.3, rot_deg=5.0)
ABLATIONS = {
    "ablate_crop_align8": dict(crop_align=8),
    "ablate_no_jpeg": dict(jpeg_p=0.0),
    "ablate_no_rescale": dict(scale_p=0.0),
    "ablate_no_rotation": dict(rot_p=0.0),
}
OUT_ROOT = ROOT / "comparison"


def train_one_epoch(model, loader, opt, sched, cfg, device, max_batches=None):
    model.train()
    tot_loss, correct, n = 0.0, 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()                      # per-step schedule

        tot_loss += loss.item() * x.shape[0]
        correct += (logits.argmax(1) == y).sum().item()
        n += x.shape[0]
    return {"loss": tot_loss / n, "acc": correct / n}


def run(name, overrides, device, smoke=False):
    run_root = OUT_ROOT / "_smoke" if smoke else OUT_ROOT
    cfg = GeoConfig(exp_name=name, run_dir=run_root, **{**BASE, **overrides, **({"epochs": 2} if smoke else {})})
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    train_dl, val_dl, val_geo_dl = build_geo_loaders(cfg)
    next(iter(train_dl))  # same as the notebook's peek batch, so the shuffle order follows model_4
    model = build_model().to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(train_dl))

    start_epoch, best_acc = 0, 0.0
    log_path = out / "log.csv"
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        if ckpt["epoch"] + 1 >= cfg.epochs:
            print(f"[{name}] already finished (best geo val {ckpt['best_acc']:.4f}), skipping")
            return
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt["best_acc"]
        print(f"[{name}] resuming at epoch {start_epoch}, best geo val {best_acc:.4f}")
    else:
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
                                    *[f"val_acc_{c}" for c in CLASS_NAMES], "val_jpeg_loss", "val_jpeg_acc",
                                    *[f"val_jpeg_acc_{c}" for c in CLASS_NAMES], "time_s"])
    print(f"[{name}] overrides {overrides} -> {out}")

    max_tr, max_val = (5, 3) if smoke else (None, None)
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_dl, opt, sched, cfg, device, max_tr)
        va = evaluate(model, val_dl, device, max_val)
        vj = evaluate(model, val_geo_dl, device, max_val)
        dt = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.2e}", f"{tr['loss']:.4f}", f"{tr['acc']:.4f}",
                                    f"{va['loss']:.4f}", f"{va['acc']:.4f}", *[f"{a:.4f}" for a in va["per_class"]],
                                    f"{vj['loss']:.4f}", f"{vj['acc']:.4f}", *[f"{a:.4f}" for a in vj["per_class"]],
                                    f"{dt:.0f}"])
        is_best = vj["acc"] > best_acc
        if is_best:
            best_acc = vj["acc"]
            save_checkpoint(out / "best.pt", model, opt, sched, epoch, best_acc, cfg)
        save_checkpoint(out / "last.pt", model, opt, sched, epoch, best_acc, cfg)

        print(f"[{name}] ep {epoch:3d} | lr {lr:.2e} | train {tr['loss']:.4f} / {tr['acc']:.4f} | "
              f"clean {va['loss']:.4f} / {va['acc']:.4f} ({' '.join(f'{a:.3f}' for a in va['per_class'])}) | "
              f"geo {vj['loss']:.4f} / {vj['acc']:.4f} ({' '.join(f'{a:.3f}' for a in vj['per_class'])}) "
              f"| {dt:.0f}s{'  *' if is_best else ''}", flush=True)
    print(f"[{name}] done, best geo val {best_acc:.4f}")


def write_summary(names):
    rows = []
    for name in ["model_4 (reference)", *names]:
        log = ROOT / "runs/model_4/log.csv" if name.startswith("model_4") else OUT_ROOT / name / "log.csv"
        if not log.exists():
            continue
        with open(log) as f:
            ep = list(csv.DictReader(f))
        if not ep:
            continue
        best = max(ep, key=lambda r: float(r["val_jpeg_acc"]))  # first epoch reaching the max, like best.pt
        rows.append({"run": name, "epochs_done": len(ep), "best_epoch": best["epoch"],
                     "best_geo_val_acc": best["val_jpeg_acc"], "geo_val_loss_at_best": best["val_jpeg_loss"],
                     "clean_val_acc_at_best": best["val_acc"], "clean_val_loss_at_best": best["val_loss"],
                     **{f"geo_acc_{c}": best[f"val_jpeg_acc_{c}"] for c in CLASS_NAMES},
                     "train_minutes": round(sum(float(r["time_s"]) for r in ep) / 60)})
    with open(OUT_ROOT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"  {r['run']:22s} | best epoch {r['best_epoch']:>2s} | "
              f"geo loss {r['geo_val_loss_at_best']} / acc {r['best_geo_val_acc']} | "
              f"clean loss {r['clean_val_loss_at_best']} / acc {r['clean_val_acc_at_best']} | {r['train_minutes']} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(ABLATIONS), help="run only these ablations")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches into comparison/_smoke")
    args = ap.parse_args()

    names = args.only or list(ABLATIONS)
    device = get_device()
    print(f"device {device}; runs: {names}")
    t0 = time.time()
    for name in names:
        run(name, ABLATIONS[name], device, smoke=args.smoke)
        print(f"elapsed {(time.time() - t0) / 3600:.2f} h")
    if not args.smoke:
        OUT_ROOT.mkdir(exist_ok=True)
        print("summary (comparison/summary.csv):")
        write_summary(list(ABLATIONS))


if __name__ == "__main__":
    main()
