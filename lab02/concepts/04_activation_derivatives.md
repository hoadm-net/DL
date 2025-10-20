# Concept: Activation Derivatives & Stability

## 🎯 Objective
Understand how to compute derivatives for common activations and handle numerical stability in code.

---

## 📖 Detailed theory

Why derivatives matter: backpropagation needs them to update parameters.

Common derivatives:
- Sigmoid: $\sigma'(x) = \sigma(x)\,[1 - \sigma(x)]$
- ReLU: $\mathbf{1}_{x>0}$ (use 0 at x = 0)
- Tanh: $1 - \tanh^2(x)$

Numerical stability tips:
- Clip inputs for extreme values (e.g., sigmoid for |x| > 40)
- Prefer `np.tanh` for tanh; it’s already stable
- For large arrays, avoid unnecessary recomputation: reuse activations for derivatives

---

## 🤔 Conceptual focus
- Backprop requires local derivatives at each node; chain rule multiplies them
- Saturating activations shrink gradients; non-saturating (ReLU) keep them larger
- Numerical stability is a practical concern, not just theory

---

## 🧪 Quick checks (conceptual)
- Can you derive or recall each derivative quickly?
- Can you explain why saturation makes learning slow?
- Can you list at least two stability tricks for implementations?

---

## ✅ Success criteria
- You know and can explain the derivative formulas
- You can reason about gradient flow qualitatively through a network
- You can articulate stability concerns and common mitigations
