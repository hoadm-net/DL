# Lab 04: Classification Fundamentals

## Objective
Understand probabilistic classification with sigmoid/softmax and cross-entropy loss for binary and multi-class problems.

## What you'll learn
- Binary classification with logistic regression (sigmoid + BCE)
- Softmax for multi-class probabilities
- Cross-entropy and negative log-likelihood (NLL)
- Stable computations (subtract-max, log-sum-exp), calibration, thresholds

## Folder layout
```
lab04/
├── README.md        # This overview
├── concepts/        # Detailed theory (no code)
└── exercises/       # You write your own practice code (optional)
```

Concept guides:
- concepts/01_binary_classification.md
- concepts/02_softmax.md
- concepts/03_cross_entropy.md
- concepts/04_softmax_ce_gradients.md

## How to use this lab
1. Read the concepts in order; connect them to Labs 01–03.
2. Draw decision boundaries and probability maps to build intuition.
3. If you implement, do it in your own files; this repo stays concepts-first.
4. Move on when you can explain key formulas and common pitfalls.

## Completion checklist
- You can derive/formulate sigmoid, softmax, and cross-entropy
- You can explain decision boundaries and thresholding for binary classification
- You understand numerical stability tricks and why they’re needed
- You can state gradient forms used in training classifiers
