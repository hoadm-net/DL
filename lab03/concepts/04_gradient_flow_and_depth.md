# Concept: Gradient Flow and Depth

## 🎯 Objective
Understand why gradients vanish or explode in deep networks and how to mitigate these issues.

---

## 📖 The problem
- Backprop multiplies many Jacobians → products of derivatives
- If average |derivative| < 1 → vanishing gradients
- If average |derivative| > 1 → exploding gradients

Formal hint (scalar chain):
$$\frac{\partial \mathcal{L}}{\partial \theta} = \prod_{\ell=1}^{L} g_\ell \quad \text{with many } g_\ell \in (0,1) \text{ or } >1$$

---

## 🧰 Mitigations
- Proper initialization (Xavier/He)
- Non-saturating activations (ReLU family)
- Normalization layers (BatchNorm/LayerNorm)
- Residual connections (ResNets) shorten gradient paths
- Gradient clipping (for RNNs and unstable phases)
- Learning rate schedules and optimizers (Adam, etc.)

---

## 🧠 Intuition
- Residual links create near-identity mappings → gradients skip long products
- Normalization keeps activations in friendly ranges layer-to-layer
- Architecture choices directly impact gradient health

---

## ✅ Success criteria
- You can explain vanishing/exploding gradients qualitatively
- You can name at least three mitigation strategies and why they help
- You can relate initialization/activation choices to gradient behavior
