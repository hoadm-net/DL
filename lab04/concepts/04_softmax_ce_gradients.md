# Concept: Gradients of Softmax + Cross-Entropy

## 🎯 Objective
Derive the gradients used to train multi-class classifiers efficiently and stably.

---

## 📖 Gradient results (from logits)
Let $\mathbf{z}$ be logits, $\mathbf{p} = \text{softmax}(\mathbf{z})$, and $\mathbf{y}$ be one-hot.

Cross-entropy: $\mathcal{L} = -\sum_k y_k \log p_k$.

Then the gradient w.r.t. logits is:
$$\frac{\partial \mathcal{L}}{\partial z_k} = p_k - y_k$$

Batch mean divides by $m$ if you average over samples.

---

## 🧠 Intuition
- The gradient pushes probability mass from wrong classes to the correct one
- Simplicity of $p - y$ is why softmax + CE is popular
- Working in logits avoids numerical issues and is easy to vectorize

---

## 🔗 Related derivatives
- For binary logistic regression (sigmoid + BCE), $\partial \mathcal{L}/\partial z = p - y$
- For weighted or label-smoothed targets, replace $\mathbf{y}$ accordingly

---

## ✅ Success criteria
- You can state $\nabla_{\mathbf{z}} \text{CE} = \mathbf{p} - \mathbf{y}$
- You can explain why it’s numerically and computationally convenient
- You can connect binary and multi-class cases via the same pattern
