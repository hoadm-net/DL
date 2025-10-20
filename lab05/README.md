# Lab 05: Optimization Algorithms

## Objective
Understand practical optimization methods for training neural networks beyond vanilla gradient descent.

## What you'll learn
- Stochastic and mini-batch gradient descent
- Momentum and Nesterov Accelerated Gradient (NAG)
- Adaptive methods (AdaGrad, RMSProp, Adam)
- Learning rate schedules and practical training tips

## Folder layout
```
lab05/
├── README.md        # This overview
├── concepts/        # Detailed theory (no code)
└── exercises/       # You write your own practice code (optional)
```

Concept guides:
- concepts/01_sgd_and_minibatch.md
- concepts/02_momentum_nesterov.md
- concepts/03_adaptive_methods_adam.md
- concepts/04_lr_schedules_and_practice.md

## How to use this lab
1. Read the concepts in order; connect updates to gradients from earlier labs.
2. Compare update rules side-by-side; track hyperparameters and roles.
3. If you implement, do so outside this repo; we focus on theory here.
4. Move on when you can select an optimizer and justify the choice.

## Completion checklist
- You can explain mini-batch SGD and its variance trade-offs
- You can write momentum/Nesterov updates and explain their effects
- You can define Adam’s moving averages and bias correction
- You can list common LR schedules and when to use them
