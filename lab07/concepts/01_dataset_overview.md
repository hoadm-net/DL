# Dataset Overview: Structure, Splits, and Pitfalls

Goal: establish a disciplined approach to handling real datasets for deep learning experiments.

## Dataset anatomy
- Features X: images, text tokens, tabular columns, etc.
- Labels y: class ids, one-hot vectors, targets.
- Metadata: sample ids, timestamps, source, folds.

## Splits
- Train / Validation / Test (hold-out)
- Ratios: common 80/10/10 or 70/15/15; depends on dataset size.
- Stratified splits: preserve class distribution across splits.
- Random seeds: fix seeds to ensure reproducibility across runs.
- K-Fold cross-validation (when data is scarce) — avoid leakage.

## Leakage risks and mitigations
- Temporal leakage: future info in training — use time-aware splits.
- Identity leakage: same subject in train and test — group-aware splits.
- Preprocessing leakage: fit scalers on full data — fit on train only.

## Data integrity checks
- Class distribution histograms per split.
- Duplicate detection (hashing) across splits.
- Sanity set: tiny subset to validate pipeline quickly.

## MNIST and CIFAR quick notes
- MNIST: grayscale 28x28, classes 0–9, common train/test provided; create your own validation.
- CIFAR-10: color 32x32; normalize per-channel using training set statistics.
