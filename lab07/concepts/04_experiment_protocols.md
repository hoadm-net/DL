# Experiment Protocols: Reproducibility and Fairness

Establish habits that make experiments trustworthy and comparable.

## Reproducibility
- Fix random seeds for all RNG sources.
- Log dataset versions, split seeds, hyperparameters, and metrics.
- Save train/val/test indices for reuse.

## Early stopping and validation
- Monitor validation metric; stop when it stops improving for N epochs.
- Keep a separate test set used only once after model selection.

## Fair comparisons
- Same splits, preprocessing, and budgets across methods.
- Hyperparameter search: define ranges and report best validation setting.

## Avoiding leakage
- Fit preprocessing on training only; apply to val/test.
- Ensure no identity/time leakage across splits.

## Result reporting
- Report central tendency and variability (mean ± std) over multiple runs.
- Provide confusion matrices and per-class metrics for transparency.
