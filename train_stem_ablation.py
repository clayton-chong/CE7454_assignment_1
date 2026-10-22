"""Stem ablation: model_4 with a standard ResNet-18 stem instead of the stride-1 stem.

    ForensicResNet (model_4) : [3x3 conv, s1] x 2           -> stages with strides (2, 2, 2, 2)  = 16x total
    ablate_std_stem          : 7x7 conv s2 -> 3x3 maxpool s2 -> stages with strides (1, 2, 2, 2)  = 32x total

Widths (32-64-128-256), depths, augmentation, optimiser, schedule, 50 epochs, seed and the geometric
validation set are identical to model_4, so the only difference is the ResNet-18 stem layout.
Training reuses train_comparison.run(); output goes to comparison/ablate_std_stem/.

    python train_stem_ablation.py --smoke             # pipeline check
    caffeinate -i python train_stem_ablation.py       # full run (~1-1.5 h), then evaluation
    python train_stem_ablation.py --eval-only         # only re-run the evaluation at the end

After training, last.pt and best.pt are evaluated on the clean / jpeg / geometric validation sets and the
share of "real" predictions on test/ is recorded (eval_std_stem.csv in the run folder).
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_comparison
from config import CLASS_NAMES, ROOT
from data import DevDataset, build_val_jpeg_loader
from geo_aug_data import GeoConfig, build_geo_loaders
from helpers import evaluate, get_device
from model import BasicBlock, ForensicResNet

NAME = "ablate_std_stem"


class StdStemResNet(ForensicResNet):
    """ForensicResNet with the ResNet-18 stem: 7x7 conv stride 2 + 3x3 max-pool stride 2, first stage stride 1."""

    def __init__(self, stem_ch: int = 32, **kw):
        super().__init__(stem_ch=stem_ch, **kw)
        self.stem = nn.Sequential(nn.Conv2d(3, stem_ch, 7, stride=2, padding=3, bias=False),
                                  nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, stride=2, padding=1))
        w0 = self.stages[0][0].conv1.out_channels
        self.stages[0][0] = BasicBlock(stem_ch, w0, stride=1)  # stem already downsampled 4x
        self._init_weights()  # same init scheme as ForensicResNet (incl. zero-init last BN per block)


def build_std_stem_model() -> nn.Module:
    return StdStemResNet()


def load_std_stem(path: Path, device) -> nn.Module:
    model = build_std_stem_model().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def evaluate_run(out: Path, device, num_workers: int = 4, name: str = NAME, load_fn=None,
                 csv_name: str = "eval_std_stem.csv"):
    load_fn = load_fn or load_std_stem
    cfg = GeoConfig(exp_name="eval", num_workers=num_workers)
    _, clean_dl, geo_dl = build_geo_loaders(cfg)
    loaders = {"clean": clean_dl, "jpeg": build_val_jpeg_loader(cfg), "geometric": geo_dl}
    test_dl = DataLoader(DevDataset(ROOT / "test"), batch_size=32, shuffle=False, num_workers=num_workers)

    rows = []
    for ck in ("best.pt", "last.pt"):
        epoch = torch.load(out / ck, map_location="cpu", weights_only=False)["epoch"]
        model = load_fn(out / ck, device)
        for set_name, dl in loaders.items():
            r = evaluate(model, dl, device)
            print(f"{name}/{ck} (ep {epoch:2d}) | {set_name:9s} | loss {r['loss']:.4f} / acc {r['acc']:.4f} "
                  f"({' '.join(f'{a:.3f}' for a in r['per_class'])})", flush=True)
            rows.append({"checkpoint": ck, "epoch": epoch, "set": set_name, "loss": f"{r['loss']:.4f}",
                         "acc": f"{r['acc']:.4f}",
                         **{f"acc_{c}": f"{a:.4f}" for c, a in zip(CLASS_NAMES, r["per_class"])}})
        with torch.no_grad():  # test labels are unknown: only the prediction distribution is recorded
            preds = torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in test_dl])
        counts = torch.bincount(preds, minlength=len(CLASS_NAMES)).tolist()
        print(f"{name}/{ck} (ep {epoch:2d}) | test predictions {dict(zip(CLASS_NAMES, counts))} | "
              f"real ratio {counts[0] / len(preds):.4f}", flush=True)
        # test row: "acc" = share predicted real; acc_<class> = share predicted as each class
        rows.append({"checkpoint": ck, "epoch": epoch, "set": "test_real_ratio", "loss": "",
                     "acc": f"{counts[0] / len(preds):.4f}",
                     **{f"acc_{c}": f"{n / len(preds):.4f}" for c, n in zip(CLASS_NAMES, counts)}})

    with open(out / csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"saved {out / csv_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches into comparison/_smoke")
    ap.add_argument("--eval-only", action="store_true", help="skip training, only evaluate the saved checkpoints")
    args = ap.parse_args()

    m = build_std_stem_model()
    with torch.no_grad():
        shape = tuple(m.stem(torch.zeros(1, 3, 256, 256)).shape)
    print(f"std-stem model: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params, stem output {shape}")

    device = get_device()
    if not args.eval_only:
        train_comparison.build_model = build_std_stem_model  # run() builds its model through this name
        train_comparison.run(NAME, {}, device, smoke=args.smoke)
    if not args.smoke:
        evaluate_run(train_comparison.OUT_ROOT / NAME, device)
        train_comparison.write_summary([*train_comparison.ABLATIONS, NAME])


if __name__ == "__main__":
    main()
