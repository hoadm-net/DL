# Concept: ReLU

## 🎯 Objective
Understand the Rectified Linear Unit (ReLU), why it’s popular, and its limitations.

---

## 📖 Detailed theory

Definition:
$$\text{ReLU}(x) = \max(0, x)$$

Key properties:
- Output range: [0, +∞)
- Non-saturating for positive inputs → stronger gradients
- Sparse activations (exact zeros for x < 0)
- Not zero-centered; risk of “dying ReLUs”

When to use: default choice in many deep networks.

---

## 🧮 Derivative
$$\text{ReLU}'(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \\ \text{undefined (use 0 or 1)} & x = 0 \end{cases}$$

Practical choice: use 0 at x = 0.

---

## 🤔 Variants and pitfalls
- Variants: Leaky ReLU, PReLU, ELU, GELU (address dying ReLU and smoothness)
- Pitfall: dying ReLU — neurons stuck at zero when gradients vanish for x < 0
- Mitigations: proper initialization, leaky variants, learning rate tuning

---

## 🧪 Quick checks (conceptual)
- For x < 0 → 0; for x > 0 → x
- Derivative: 0 for x <= 0, 1 for x > 0 (choose 0 at x = 0)
- Expect sparsity (many exact zeros) when inputs are negative

---

## ✅ Success criteria
- You can explain why ReLU accelerates training vs sigmoid/tanh
- You can state the definition and derivative
- You understand dying ReLU and common mitigations
