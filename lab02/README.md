# Lab 02: Activation Functions

## Objective
Build a solid conceptual understanding of activation functions and how they enable neural networks to learn non-linear relationships.

## What you'll learn
- Why non-linearity is essential beyond linear models
- Sigmoid, ReLU, Tanh: behavior, intuition, and use cases
- Activation derivatives and their role in backpropagation
- Practical considerations: saturation, dying neurons, vanishing gradients

## Folder layout
```
lab02/
├── README.md        # This overview
├── concepts/        # Detailed explanations and small tasks
└── exercises/       # Your practice implementations
```

Concept guides (concepts-first, you will write the code yourself):
- concepts/01_sigmoid.md
- concepts/02_relu.md
- concepts/03_tanh.md
- concepts/04_activation_derivatives.md

## How to use this lab
1. Read the concept notes in order (1 → 4), focusing on intuition first.
2. Write your own implementations in a place you prefer (this repo does not require code).
3. Test with simple inputs and edge cases you design.
4. Move on when you can explain both the function and its derivative succinctly.

## Completion checklist
- Explain why non-linearity is necessary in a sentence or two
- Describe properties, pros/cons, and typical uses for sigmoid, ReLU, and tanh
- Derive or state the derivatives and explain how they are used in backprop
- Identify risks (saturation, dying ReLU) and mitigation ideas at a high level
