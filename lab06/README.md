# Lab 06: Regularization

## Objective
Control model complexity to reduce overfitting and improve generalization.

## What you'll learn
- Overfitting: causes, symptoms, and capacity control
- L2/L1 regularization and weight decay
- Dropout and stochastic regularizers
- Practical regularization toolbox (early stopping, data augmentation, etc.)

## Folder layout
```
lab06/
├── README.md        # This overview
├── concepts/        # Detailed theory (no code)
└── exercises/       # You write your own practice code (optional)
```

Concept guides:
- concepts/01_overfitting_and_capacity.md
- concepts/02_l2_l1_regularization.md
- concepts/03_dropout_and_stochastic_regularizers.md
- concepts/04_practical_regularization.md

## How to use this lab
1. Read the concepts in order; connect to bias–variance trade-off.
2. Analyze learning curves to identify under/overfitting.
3. If you implement, do it outside this repo; we focus on theory here.
4. Move on when you can justify regularization choices for a task.

## Completion checklist
- You can explain overfitting and capacity with clear examples
- You can write L2/L1 penalty terms and relate weight decay to L2
- You understand dropout’s effect on training/inference
- You can outline a regularization plan for a new dataset
