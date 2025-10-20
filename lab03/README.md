# Lab 03: Multi-layer Networks

## Objective
Understand how stacking layers creates powerful non-linear models and how forward/backward passes work in deep networks.

## What you'll learn
- Layer composition: affine (linear) + activation blocks
- Forward pass through multiple layers (shapes, dimensions)
- Backpropagation through stacked layers (chain rule at scale)
- Initialization and gradient flow (vanishing/exploding)

## Folder layout
```
lab03/
├── README.md        # This overview
├── concepts/        # Detailed theory (no code)
└── exercises/       # You write your own practice code (optional)
```

Concept guides:
- concepts/01_layer_composition.md
- concepts/02_forward_backward_multilayer.md
- concepts/03_weight_initialization.md
- concepts/04_gradient_flow_and_depth.md

## How to use this lab
1. Read the concepts in order; focus on shapes and the chain rule.
2. Draw computation graphs by hand to follow gradient paths.
3. If you implement, do it in your own files; this repo stays concepts-first.
4. Move on when you can explain each formula and its intuition.

## Completion checklist
- You can describe a multi-layer network as repeated (affine + activation) blocks
- You can write down forward and backward equations layer by layer
- You understand Xavier/He initialization and when to use each
- You can explain vanishing/exploding gradients and common mitigations
