# Concept: Binary Classification (Logistic Regression)

## 🎯 Objective
Model binary outcomes with a probabilistic output using the sigmoid function and binary cross-entropy (BCE).

---

## 📖 Model
Given features $\mathbf{x} \in \mathbb{R}^n$ and label $y \in \{0,1\}$:

- Logit (score): $z = \mathbf{w}^\top \mathbf{x} + b$
- Probability: $p(y=1\mid \mathbf{x}) = \sigma(z) = \tfrac{1}{1+e^{-z}}$

Decision rule (threshold $\tau$): predict class 1 if $p \ge \tau$, else 0 (common choice: $\tau=0.5$).

---

## 🧮 Loss: Binary Cross-Entropy (BCE)

For one sample:
$$\mathcal{L}(p,y) = -\big[y\,\log p + (1-y)\,\log(1-p)\big]$$

For a batch (mean):
$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\Big]$$

---

## 🧠 Intuition
- BCE penalizes confident wrong predictions heavily (log penalty)
- Thresholding turns probabilities into hard decisions; ROC/PR curves trade off thresholds
- Calibration: probabilities should reflect frequencies (e.g., 0.8 ≈ 80% positive)

---

## ⚠️ Stability
- Clamp $p$ away from 0 and 1 in practice (e.g., $[\epsilon, 1-\epsilon]$)
- Or compute BCE from logits directly to avoid $\log(0)$ issues

---

## ✅ Success criteria
- You can write logistic regression with BCE and explain thresholding
- You can reason about calibration and decision metrics
- You can explain why logs make BCE sensitive to confident mistakes
