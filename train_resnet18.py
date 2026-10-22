"""Baseline: a full-width ResNet-18, trained from scratch with model_4's full recipe.

Answers "why not just use ResNet-18?". Architecture follows He et al. (2016) for 3 classes:

    stem   : 7x7 conv, 64, stride 2 -> 3x3 max-pool, stride 2
    stages : 64-128-256-512 channels, 2 BasicBlocks each, strides (1, 2, 2, 2)  -> 32x total downsampling
    head   : global average pool -> dropout 0.2 -> fc(512 -> 3)

Everything else is identical to model_4: all augmentations (rescale, rotation, JPEG re-save, unaligned
crops, flips), AdamW 1e-3 + warm-up/cosine, 50 epochs, weight decay, label smoothing, dropout 0.2,
zero-init last BN per block, seed, split, and best.pt chosen on the geometric validation set.
No pretrained weights (course rules). Output: comparison/resnet18_full/.

    python train_resnet18.py --smoke             # pipeline check + timing of a few batches
    caffeinate -i python train_resnet18.py       # full run (~1-1.5 h estimated), then evaluation
    python train_resnet18.py --eval-only         # only re-run the evaluation at the end
"""
import argparse

import torch
import torch.nn as nn

import train_comparison
from helpers import get_device
from train_stem_ablation import StdStemResNet, evaluate_run

NAME = "resnet18_full"
PARAM_LIMIT = 12_300_000


def build_resnet18() -> nn.Module:
    return StdStemResNet(stem_ch=64, widths=(64, 128, 256, 512), depths=(2, 2, 2, 2))


def load_resnet18(path, device) -> nn.Module:
    model = build_resnet18().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches into comparison/_smoke")
    ap.add_argument("--eval-only", action="store_true", help="skip training, only evaluate the saved checkpoints")
    args = ap.parse_args()

    m = build_resnet18()
    n_params = sum(p.numel() for p in m.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} params exceeds the {PARAM_LIMIT:,} limit"
    print(f"ResNet-18: {n_params / 1e6:.2f}M params (limit {PARAM_LIMIT / 1e6:.1f}M)")

    device = get_device()
    if not args.eval_only:
        train_comparison.build_model = build_resnet18  # run() builds its model through this name
        train_comparison.run(NAME, {}, device, smoke=args.smoke)
    if not args.smoke:
        evaluate_run(train_comparison.OUT_ROOT / NAME, device, name=NAME, load_fn=load_resnet18,
                     csv_name="eval_resnet18.csv")
        train_comparison.write_summary([*train_comparison.ABLATIONS, "ablate_std_stem", NAME])


if __name__ == "__main__":
    main()
