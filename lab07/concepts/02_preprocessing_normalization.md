# Preprocessing and Normalization

Reliable pipelines separate fitting (compute statistics) from transforming.

## Scaling and standardization
- Min-Max scaling: x' = (x - min_train) / (max_train - min_train)
- Standardization: x' = (x - μ_train) / σ_train
- For images: per-channel mean/std computed on training set.

## Categorical targets
- One-hot encoding for K classes: y ∈ {0..K-1} → e_k ∈ R^K
- Label smoothing (for soft targets): y_smooth = (1-ε)·e_k + ε/K

## Batching and shuffling
- Shuffle training data every epoch; do not shuffle validation/test.
- Batch size trade-offs: stability vs. generalization.

## Data augmentation (conceptual)
- Only on training; keep validation/test distributions clean.
- For images: flips, crops, color jitter; beware label-preserving constraints.

## Numerical stability
- Normalize before feeding into models to improve conditioning.
- Keep transformations deterministic given a seed for reproducibility.
