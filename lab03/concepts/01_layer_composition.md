# Concept: Layer Composition (Blocks)

## 🎯 Objective
Understand the building block of deep networks: alternating affine (linear) transformations and non-linear activations.

---

## 📖 Detailed theory

A typical layer applies an affine transformation followed by a non-linearity:
$$\mathbf{h} = f(\mathbf{W}\,\mathbf{x} + \mathbf{b})$$

- Input (column vector): $\mathbf{x} \in \mathbb{R}^{n}$
- Weights: $\mathbf{W} \in \mathbb{R}^{m\times n}$
- Bias: $\mathbf{b} \in \mathbb{R}^{m}$
- Activation: $f$ (e.g., ReLU, Tanh, Sigmoid)
- Output: $\mathbf{h} \in \mathbb{R}^{m}$

Stacking $L$ such blocks yields a deep network:
$$\mathbf{h}^{(\ell)} = f^{(\ell)}\!\big(\mathbf{W}^{(\ell)}\,\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\big),\quad \ell=1,\dots,L$$
with $\mathbf{h}^{(0)}=\mathbf{x}$.

---

## 🧠 Intuition
- Affine parts mix and rescale features; activations add non-linearity.
- Depth increases expressivity by composing simple functions.
- Without activations, stacked affine maps collapse to a single affine map (no added power).

---

## 🔎 Shape discipline (batch form)
For a batch of $m$ samples:
- $\mathbf{X} \in \mathbb{R}^{m\times n}$
- $\mathbf{H}^{(\ell)} = f\big(\mathbf{X}^{(\ell)}\,\mathbf{W}^{(\ell)} + \mathbf{1}\,\mathbf{b}^{(\ell)\top}\big)$ where $\mathbf{X}^{(\ell)}\equiv\mathbf{H}^{(\ell-1)}$
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{n_{\ell}\times n_{\ell-1}}$, $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_{\ell}}$
- $\mathbf{H}^{(\ell)} \in \mathbb{R}^{m\times n_{\ell}}$

Broadcasting note: adding $\mathbf{b}^{(\ell)}$ to each row of the pre-activation.

---

## ✅ Success criteria
- You can define a layer as (affine + activation)
- You can track shapes through multiple layers (single sample and batch)
- You can explain why activations are necessary in deep networks
