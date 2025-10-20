# Evaluation Metrics

Accuracy is not enough, especially for imbalanced datasets.

## Confusion-matrix derived metrics
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2·Precision·Recall / (Precision + Recall)
- Macro vs. micro averaging: macro averages per-class metrics; micro aggregates globally.

## ROC and AUC
- ROC curve plots TPR vs. FPR as threshold varies.
- AUC summarizes ROC into a single scalar; threshold-invariant.
- For severely imbalanced data, PR curves can be more informative.

## Calibration
- Well-calibrated probabilities match empirical frequencies.
- Techniques: temperature scaling, isotonic regression (conceptual overview only).

## Reporting
- Always report: accuracy, macro-F1 (for class imbalance), confusion matrix.
- Include mean ± std across multiple seeds/splits.
