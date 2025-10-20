# Concept: Tanh

## 🎯 Objective
Understand the hyperbolic tangent activation, when it helps, and its downsides.

---

## 📖 Detailed theory

Definition:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Key properties:
- Output range: (-1, 1)
- Zero-centered (advantage over sigmoid)
- Smooth and differentiable
- Saturates for large |x| → vanishing gradients

---

## 🧮 Derivative
$$\tanh'(x) = 1 - \tanh^2(x)$$

---

## 🤔 When to use
- Hidden layers when zero-centered activations help (optimization)
- RNNs historically used tanh (before modern variants)
- Avoid deep stacks without care: still suffers from saturation

---

## 🧪 Quick checks (conceptual)
- For x = 0 → 0
- For large positive x → ~1; large negative x → ~-1
- Odd symmetry: $\tanh(-x) = -\tanh(x)$

---

## ✅ Success criteria
- You can state the definition and derivative
- You can contrast tanh vs sigmoid (zero-centered vs not)
- You can explain saturation and its effect on gradients
