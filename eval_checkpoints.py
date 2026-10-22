"""Evaluate trained checkpoints on all three validation sets (same 1,500 images, different conditions):

    clean     : original files (quality 95, 4:4:4)
    jpeg      : re-saved at quality 90, 4:2:0, no geometric change   (splits/val_q90_sub2)
    geometric : fixed rescale / rotation / off-grid crop, then q90 4:2:0 (splits/val_geo_q90_sub2)

    python eval_checkpoints.py                                   # model_1, model_2, model_4 last.pt
    python eval_checkpoints.py --ckpts runs/model_2/best.pt comparison/ablate_no_jpeg/best.pt

Prints one line per checkpoint and set, plus confusion matrices with --cm, and writes eval_summary.csv.
"""
import argparse
import csv
from pathlib import Path

import torch

from config import CLASS_NAMES, ROOT
from data import build_val_jpeg_loader
from geo_aug_data import GeoConfig, build_geo_loaders
from helpers import evaluate, format_cm, get_device, load_model

DEFAULT_CKPTS = ["runs/model_1/last.pt", "runs/model_2/last.pt", "runs/model_4/last.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=DEFAULT_CKPTS, help="checkpoint paths (relative to the project)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cm", action="store_true", help="also print confusion matrices")
    ap.add_argument("--out", type=Path, default=ROOT / "eval_summary.csv")
    args = ap.parse_args()

    device = get_device()
    cfg = GeoConfig(exp_name="eval", num_workers=args.num_workers)
    _, clean_dl, geo_dl = build_geo_loaders(cfg)
    loaders = {"clean": clean_dl, "jpeg": build_val_jpeg_loader(cfg), "geometric": geo_dl}

    rows = []
    for ck in args.ckpts:
        path = ROOT / ck
        model, _ = load_model(path, device)
        epoch = torch.load(path, map_location="cpu", weights_only=False)["epoch"]
        for set_name, dl in loaders.items():
            r = evaluate(model, dl, device)
            print(f"{ck:40s} (ep {epoch:2d}) | {set_name:9s} | loss {r['loss']:.4f} / acc {r['acc']:.4f} "
                  f"({' '.join(f'{a:.3f}' for a in r['per_class'])})", flush=True)
            if args.cm:
                print(format_cm(r["cm"]))
            rows.append({"checkpoint": ck, "epoch": epoch, "val_set": set_name,
                         "loss": f"{r['loss']:.4f}", "acc": f"{r['acc']:.4f}",
                         **{f"acc_{c}": f"{a:.4f}" for c, a in zip(CLASS_NAMES, r["per_class"])}})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
