# Concept: Forward and Backward in Multi-layer Networks

## 🎯 Objective
Understand forward propagation and backpropagation across multiple layers, keeping a strict handle on shapes and the chain rule.

---

## 📖 Forward pass (L layers)
Given input $\mathbf{x}=\mathbf{h}^{(0)}$:
1. Pre-activation: $\mathbf{z}^{(\ell)} = \mathbf{h}^{(\ell-1)}\,\mathbf{W}^{(\ell)} + \mathbf{b}^{(\ell)}$
2. Activation: $\mathbf{h}^{(\ell)} = f^{(\ell)}(\mathbf{z}^{(\ell)})$

For batches (rows are samples):
- $\mathbf{Z}^{(\ell)} = \mathbf{H}^{(\ell-1)}\,\mathbf{W}^{(\ell)} + \mathbf{1}\,\mathbf{b}^{(\ell)\top}$
- $\mathbf{H}^{(\ell)} = f^{(\ell)}(\mathbf{Z}^{(\ell)})$

---

## 🔁 Backward pass (chain rule)
Let $\mathcal{L}$ be the loss on the batch. Define $\delta^{(\ell)} = \partial \mathcal{L}/\partial \mathbf{z}^{(\ell)}$.

From output to input:
1. $\delta^{(L)} = (\partial \mathcal{L}/\partial \mathbf{h}^{(L)}) \odot f'^{(L)}(\mathbf{z}^{(L)})$
2. For $\ell=L-1,\dots,1$:
   $$\delta^{(\ell)} = \big(\delta^{(\ell+1)} (\mathbf{W}^{(\ell+1)})^{\top}\big) \odot f'^{(\ell)}(\mathbf{z}^{(\ell)})$$

Parameter gradients:
- $\partial \mathcal{L}/\partial \mathbf{W}^{(\ell)} = (\mathbf{H}^{(\ell-1)})^{\top} \delta^{(\ell)}$
- $\partial \mathcal{L}/\partial \mathbf{b}^{(\ell)} = \text{rowsum}(\delta^{(\ell)})$

Batch averaging: divide by batch size if using mean loss.

---

## 🧠 Intuition
- Each layer contributes a local gradient; backprop multiplies them along the path.
- Storing $\mathbf{z}^{(\ell)}$ and $\mathbf{h}^{(\ell)}$ during the forward pass makes backward efficient.
- Shape sanity checks catch most bugs (mismatched matrix dimensions).

---

## ✅ Success criteria
- You can write forward and backward equations for each layer
- You can explain the role of $\delta^{(\ell)}$ and how it propagates
- You can derive parameter gradients and bias gradients in batch form
