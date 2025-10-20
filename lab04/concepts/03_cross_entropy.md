# Concept: Cross-Entropy and NLL

## 🎯 Objective
Understand cross-entropy as a measure of distance between the true distribution and predicted probabilities.

---

## 📖 Definitions

For one-hot target $\mathbf{y} \in \{0,1\}^K$ and predicted probs $\mathbf{p}$:
$$\text{CE}(\mathbf{y}, \mathbf{p}) = -\sum_{k=1}^{K} y_k \log p_k$$

Binary case (BCE) is a special case with $K=2$.

Negative log-likelihood (NLL) with class index $t$:
$$\text{NLL}(t, \mathbf{p}) = -\log p_t$$

Batch mean over $m$ samples:
$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} -\log p_{t^{(i)}}^{(i)}$$

---

## 🧠 Intuition
- Penalizes assigning low probability to the true class
- Encourages calibrated, confident predictions on the correct class
- Log function creates large penalties for very wrong, confident predictions

---

## ⚠️ Stability: log-sum-exp (from logits)
Compute CE directly from logits $\mathbf{z}$ without explicit softmax:
$$\text{CE}(t, \mathbf{z}) = -z_t + \log\Big(\sum_{k} e^{z_k}\Big)$$
Use subtract-max inside the log-sum-exp to avoid overflow.

---

## ✅ Success criteria
- You can define CE and NLL and explain the relationship
- You can compute CE from logits using log-sum-exp
- You understand why CE/NLL is the standard loss for classification
