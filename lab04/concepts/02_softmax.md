# Concept: Softmax for Multi-class Classification

## 🎯 Objective
Convert scores (logits) into probabilities over K classes while preserving relative differences.

---

## 📖 Definition
Given logits $\mathbf{z} \in \mathbb{R}^K$:
$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Properties:
- Range: $(0,1)$ and sums to 1
- Order preserving: increasing one logit increases its probability
- Translation invariant: adding a constant to all logits doesn’t change probs

---

## 🧠 Intuition
- Exponentiation magnifies differences → sharper distributions
- Softmax turns scores into a categorical distribution over classes

---

## ⚠️ Stability: subtract-max trick
Compute with $\tilde{z}_k = z_k - \max_j z_j$:
$$\text{softmax}(\mathbf{z})_k = \frac{e^{\tilde{z}_k}}{\sum_j e^{\tilde{z}_j}}$$
This prevents overflow when logits are large.

---

## ✅ Success criteria
- You can define softmax and list its key properties
- You can explain the subtract-max trick and why it’s safe
- You can interpret softmax outputs as probabilities
