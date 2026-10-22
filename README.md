# CE7454 Project 1: Real vs. AI-Generated Image Attribution

Code for a 3-class image attribution model (0 = real, 1 = Model A, 2 = Model B), trained from scratch.
The final model, **ForensicResNet** (2.8M parameters), reached **0.754 accuracy on the test set**.
Details and analysis are in the accompanying report.

## Repository layout

| Path | Contents |
|---|---|
| `config.py` | Default settings (`Config`) |
| `data.py` | Train/validation split, base dataset, JPEG-only validation set |
| `geo_aug_data.py` | Geometric + JPEG augmentation (`GeoConfig`) and the geometric validation set |
| `model.py` | ForensicResNet architecture |
| `helpers.py` | Optimiser, LR schedule, evaluation, checkpoint saving and loading |
| `train.ipynb` | Training notebook for the final model (`model_4`) |
| `predict_test.py` | Writes test-set predictions (`predictions.json`) |
| `eval_checkpoints.py` | Scores checkpoints on the clean, JPEG-only and geometric validation sets |
| `train_comparison.py` | Augmentation ablations (one augmentation removed per run) |
| `train_stem_ablation.py` | Ablation: ForensicResNet with a ResNet-18 stem |
| `train_resnet18.py` | Baseline: full-width ResNet-18 with the same training recipe |
| `runs/`, `comparison/` | `config.json` and `log.csv` of every run reported, plus evaluation CSVs |
| `splits/split_seed0_val500.json` | The fixed train/validation split used by all runs |
| `submissions/predictions.json` | The submitted test-set predictions |

Model checkpoints (`*.pt`) and the dataset are **not** included (see below).

## Setup

**Requirements:** Python 3.14 with the following packages (versions used for the report):

```bash
pip install torch==2.14.0 numpy==2.4.6 Pillow==12.3.0 matplotlib==3.11.0
```

Jupyter is needed for `train.ipynb`. Training uses an Apple-silicon GPU (MPS) or a CUDA GPU when one is
available and falls back to the CPU otherwise. Pillow's JPEG encoder produces the augmentation and the cached
validation sets, so a different Pillow version can change results slightly.

**Data:** the dataset is course material and is not redistributed here. Place the images in the repository
root as follows:

```
train/   15,000 images named <id>_<label>.jpg   (label 0 = real, 1 = Model A, 2 = Model B)
dev/        900 images named <id>.jpg           (optional, development phase only)
test/     1,500 images named <id>.jpg
```

The train/validation split (13,500 / 1,500, balanced, seed 0) is read from `splits/split_seed0_val500.json`, so
every run uses the same images. The JPEG-only and geometric validation sets are generated in `splits/` on first
use.

## 1. Train the final model

Open `train.ipynb` and run all cells. The config cell trains `model_4`:

```python
cfg = GeoConfig(exp_name="model_4", crop_size=256, epochs=50,
                jpeg_p=0.7, jpeg_q=(80, 95), jpeg_subsampling=(0, 2))
```

The geometric augmentation settings (`crop_align=1`, `scale_p=0.5`, `rot_p=0.3`, …) come from the defaults in
`GeoConfig`. Set `SMOKE = True` first for a quick pipeline check (a few batches per epoch).

Outputs go to `runs/model_4/`: `config.json`, `log.csv`, `best.pt` (chosen on geometric validation accuracy)
and `last.pt`. The last cells print and save the confusion matrices (`confusion_matrix.png`).

Training takes about **95 minutes** on an Apple M4 Pro (MPS). An interrupted run can be resumed by setting
`RESUME = True` with the same `exp_name`.

## 2. Predict the test set

```bash
python predict_test.py
```

This loads `runs/model_4/best.pt`, runs one forward pass per full 512×512 test image (no test-time augmentation,
no ensembling), and writes `testing/predictions.json`. The submitted predictions were produced this way from
epoch 45 (`best.pt`), giving the class counts {real: 731, Model A: 335, Model B: 434}. They are saved in
`submissions/predictions.json`.

Use `--ckpt`, `--img-dir` and `--out-dir` to change the checkpoint, the input folder or the output folder.

## 3. Reproduce the comparison results (report Table 2)

All runs below use the same training recipe as `model_4` (seed, split, schedule, 50 epochs) and differ only in the
stated setting. Each script prints one line per epoch, resumes from `last.pt` if interrupted, and skips runs
that have already finished.

**Augmentation ablations** (four runs, about 6.5 h in total):

```bash
python train_comparison.py --smoke    # quick pipeline check, writes to comparison/_smoke
python train_comparison.py            # ablate_crop_align8, ablate_no_jpeg, ablate_no_rescale, ablate_no_rotation
```

**Architecture comparisons:**

```bash
python train_stem_ablation.py         # ForensicResNet with ResNet-18 stem (about 25 min)
python train_resnet18.py              # full-width ResNet-18 baseline (about 60 min)
```

These two scripts also evaluate their checkpoints on all validation sets and record the test-set class
distribution (`eval_std_stem.csv`, `eval_resnet18.csv`). `comparison/summary.csv` collects the best epoch of
every run.

**Old model and JPEG-only model.** These are trained in `train.ipynb` with the base `Config` and data loaders
from `data.py` (8-px aligned crops, no geometric augmentation):

```python
# config cell
cfg = Config(exp_name="model_1", crop_size=256, epochs=50)                     # old model: no augmentation
cfg = Config(exp_name="model_2", crop_size=256, epochs=50,                     # JPEG re-save only
             jpeg_p=0.7, jpeg_q=(80, 95), jpeg_subsampling=(0, 2))

# data cell (in place of build_geo_loaders)
train_dl, val_dl = build_loaders(cfg)
val_jpeg_dl = build_val_jpeg_loader(cfg)
```

**Scoring on all three validation sets:**

```bash
python eval_checkpoints.py --ckpts runs/model_1/last.pt runs/model_2/last.pt runs/model_4/last.pt
```

The table reports clean and geometric validation accuracy as the mean ± standard deviation over epochs 46–50,
read from each run's `log.csv`. In the logs, `val_acc` is the clean validation set. For `model_4` and the
`comparison/` runs, `val_jpeg_acc` is the geometric validation set; for `model_2` it is the JPEG-only set. The
old and JPEG-only models are reported at the last epoch using `eval_checkpoints.py`.

```bash
python -c "
import csv, statistics as st, sys
r = list(csv.DictReader(open(sys.argv[1])))[-5:]
for k in ('val_acc', 'val_jpeg_acc'):
    v = [float(x[k]) for x in r]; print(k, f'{st.mean(v):.4f} ± {st.stdev(v):.4f}')
" runs/model_4/log.csv
```

## Notes on reproducibility

- Epochs are numbered from 0 in the code and logs, and from 1 in the report (log epoch 44 = report epoch 45).
- Seeds are fixed (training seed 42, split seed 0), but GPU kernels (MPS/CUDA) are not bit-deterministic, and
  resumed runs reshuffle the data. Expect small run-to-run differences (around ±1 percentage point).
- Since the original `model_1` run, `train.ipynb` saves `best.pt` based on the JPEG/geometric validation set
  rather than the clean set. `last.pt`, which the report uses for the old model, is unaffected.
- The model was trained from scratch: no pretrained weights, no external data and a single model (no ensembles).
