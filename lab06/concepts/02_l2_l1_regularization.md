# Concept: L2/L1 Regularization and Weight Decay

## 🎯 Objective
Use penalty terms to control weight magnitudes and reduce overfitting.

---

## 📖 Penalty terms
- L2 (ridge): $\lambda \lVert \mathbf{W} \rVert_2^2 = \lambda \sum_{i} W_i^2$
- L1 (lasso): $\lambda \lVert \mathbf{W} \rVert_1 = \lambda \sum_{i} |W_i|$

Total loss with penalty:
$$\mathcal{J}(\theta) = \mathcal{L}(\theta; \mathcal{D}) + \lambda \, \Omega(\theta)$$

---

## 🔁 Gradients and updates
- L2 gradient: $\partial \mathcal{J}/\partial W_i = \partial \mathcal{L}/\partial W_i + 2\lambda W_i$
- L1 subgradient: $\partial \mathcal{J}/\partial W_i = \partial \mathcal{L}/\partial W_i + \lambda\, \text{sign}(W_i)$ (not differentiable at 0)

Weight decay (SGD):
$$\theta_{t+1} = (1-\eta\lambda)\,\theta_t - \eta\,\nabla_\theta\mathcal{L}(\theta_t)$$
For AdamW-style decoupled decay, the $(1-\eta\lambda)\theta$ factor is applied separately from gradient updates.

---

## 🧠 Intuition and effects
- L2: shrinks weights smoothly; discourages large magnitudes
- L1: encourages sparsity (many exact zeros)
- Decoupled weight decay (AdamW) improves behavior for adaptive optimizers

---

## ✅ Success criteria
- You can write L2/L1 penalties and their gradients
- You can explain weight decay vs. L2 penalty equivalence in SGD
- You know when to prefer L1 vs L2
