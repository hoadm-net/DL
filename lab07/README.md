# Lab 07: Real Datasets (Concepts-Only)

Learn how to work with real-world datasets in a clean, reproducible way: dataset structure, train/val/test splits, preprocessing and normalization, and proper evaluation/experiment protocols. No code is provided here—students implement their own loaders and pipelines externally.

## What you will learn
- Understanding datasets and splits (train/val/test, stratification, randomness)
- Preprocessing and normalization (scaling, standardization, one-hot, batching)
- Evaluation metrics beyond accuracy (precision/recall/F1, ROC-AUC)
- Sound experiment protocols (reproducibility, early stopping, avoiding leakage)

## Concepts
- 01_dataset_overview.md — Datasets, formats, splits, and pitfalls
- 02_preprocessing_normalization.md — Scaling, standardization, batching
- 03_evaluation_metrics.md — Accuracy, precision/recall/F1, ROC-AUC
- 04_experiment_protocols.md — Reproducible and fair evaluation

## Exercises (no code in repo)
- Design a data pipeline for MNIST or CIFAR-10 with: stratified train/val/test split, normalization computed on training only, and batched iteration.
- Implement metrics: accuracy, per-class precision/recall, macro/micro F1. Compare for imbalanced toy subsets.
- Run controlled experiments with fixed seeds and early stopping. Report mean ± std over multiple runs.

Refer to the concept notes in `concepts/` for math and decision checklists.
