# Concept: Sigmoid

## 🎯 Objective
Understand the sigmoid activation, its properties, where it is useful, and its limitations.

---

## 📖 Detailed theory

Sigmoid maps inputs to (0, 1):
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Key properties:
- Output range: (0, 1)
- Centered at 0.5 (not zero-centered)
- Smooth and differentiable everywhere
- Saturates for large |x| → vanishing gradients

Intuition: a soft on/off switch that compresses any real number to a probability-like value.

---

## 🧮 Derivative
$$\sigma'(x) = \sigma(x)\,[1 - \sigma(x)]$$

Implication: small gradients when $\sigma(x)$ is near 0 or 1 (saturation).

---

## 🤔 When to use (and when not)
- Use for: output layer in binary classification (probabilities)
- Avoid for: deep hidden layers (vanishing gradients), prefer ReLU-family
- Beware: not zero-centered → can slow optimization

---

## 🧪 Quick checks (conceptual)
- For x = 0 → 0.5
- For large positive x → ~1; large negative x → ~0
- Symmetry: $\sigma(-x) = 1 - \sigma(x)$

---

## ✅ Success criteria
- You can state the formula and derivative from memory
- You can explain saturation and its training impact
- You know typical use-cases and pitfalls
