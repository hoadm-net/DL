# Lab 01: Fundamental Concepts

## Objective
Learn the four core ideas that power neural networks without diving into heavy math.

## What you'll learn
- Forward propagation — how inputs flow through a model to produce predictions
- Loss functions — how we measure how far predictions are from the targets
- Backward propagation — how learning signals (gradients) are computed
- Gradient descent — how parameters get updated to improve the model

## Folder layout

```
lab01/
├── README.md       # This overview
├── concepts/       # Detailed explanations and small tasks
└── exercises/      # Your practice scripts/notebooks
```

Concept guides:
- concepts/01_forward_propagation.md
- concepts/02_loss_functions.md
- concepts/03_backward_propagation.md
- concepts/04_gradient_descent.md

## How to use this lab
1. Read the concept files in order (1 → 4).
2. After each concept, create a small practice file in `exercises/` and implement what you learned using Python + NumPy.
3. Keep your implementations simple and readable. No external DL frameworks.
4. Move on when you can explain the concept in your own words and reproduce it in code.

## Completion checklist
- You can briefly explain each concept (one or two sentences each).
- You have small working code snippets that run forward passes, compute a loss, get gradients, and perform simple parameter updates.
- All content is written in English and kept high-level (no deep math derivations here).

When you're ready, start with `concepts/01_forward_propagation.md`.
