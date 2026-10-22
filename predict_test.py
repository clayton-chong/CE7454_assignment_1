"""Test-phase prediction: load the trained checkpoint, predict every image in test/, write testing/predictions.json.

Single model, single forward pass on the full 512x512 image (no ensembles, no TTA).
Output maps "<filename>.jpg" -> label (0 = real, 1 = Model A, 2 = Model B).
The defaults reproduce the submitted predictions (runs/model_4/best.pt, epoch 45).

    python predict_test.py
    python predict_test.py --ckpt runs/model_1/best.pt --check-val
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import CLASS_NAMES, NUM_CLASSES, ROOT
from data import DevDataset, build_loaders
from helpers import evaluate, format_cm, get_device, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "runs/model_4/best.pt")
    ap.add_argument("--img-dir", type=Path, default=ROOT / "test")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "testing")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--check-val", action="store_true", help="re-evaluate the checkpoint on the val split first")
    args = ap.parse_args()

    # 1. Data
    img_dir = args.img_dir
    n_imgs = len(list(img_dir.glob("*.jpg")))
    assert n_imgs > 0, f"no .jpg images found in {img_dir}"
    print(f"{n_imgs} test images in {img_dir}")

    # 2. Model
    device = get_device()
    model, cfg = load_model(args.ckpt, device)
    ckpt_epoch = torch.load(args.ckpt, map_location="cpu", weights_only=False)["epoch"]
    print(f"loaded {args.ckpt} (epoch {ckpt_epoch}) on {device}")

    if args.check_val:
        _, val_dl = build_loaders(cfg)
        va = evaluate(model, val_dl, device)
        print(f"val acc {va['acc']:.4f}  val loss {va['loss']:.4f}")
        print(format_cm(va["cm"]))

    # 3. Predict
    dl = DataLoader(DevDataset(img_dir), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)
    preds = {}
    model.eval()
    with torch.no_grad():
        for x, ids in dl:
            p = model(x.to(device)).argmax(1).cpu()
            for i, label in zip(ids.tolist(), p.tolist()):
                preds[f"{i}.jpg"] = int(label)

    # 4. Checks and save
    files = {p.name for p in img_dir.glob("*.jpg")}
    assert set(preds) == files, "prediction keys do not match the image filenames"
    assert all(v in range(NUM_CLASSES) for v in preds.values())

    counts = torch.bincount(torch.tensor(list(preds.values())), minlength=NUM_CLASSES).tolist()
    print("prediction counts:", dict(zip(CLASS_NAMES, counts)))

    preds = dict(sorted(preds.items(), key=lambda kv: int(Path(kv[0]).stem)))  # 0.jpg, 1.jpg, ...
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "predictions.json"
    out_path.write_text(json.dumps(preds, indent=2))
    print(f"saved {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()
