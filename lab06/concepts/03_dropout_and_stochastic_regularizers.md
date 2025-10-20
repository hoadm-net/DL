# Concept: Dropout and Stochastic Regularizers

## 🎯 Objective
Understand dropout and related stochastic techniques that regularize by injecting noise.

---

## 📖 Dropout
Randomly drop units during training with keep probability $p$:
- Train: $\tilde{h} = \mathbf{m} \odot \mathbf{h}$ with $m_i \sim \text{Bernoulli}(p)$
- Inverted dropout scales at train time: $\tilde{h} = \frac{\mathbf{m}}{p} \odot \mathbf{h}$ so no scaling needed at test
- Test: use full network (no dropout)

Effects:
- Prevents co-adaptation of features
- Acts like ensemble averaging of subnetworks

---

## 📦 Other stochastic regularizers
- Gaussian noise on activations or inputs
- Stochastic depth / drop-path (skip residual blocks randomly)
- Label smoothing (regularizes soft targets)

---

## ⚖️ Practical notes
- Tune keep probability (e.g., p=0.5 for hidden layers, higher near input)
- Combine with BatchNorm carefully (order matters)
- Use smaller dropout with modern architectures if BatchNorm/ResNet already stabilize

---

## ✅ Success criteria
- You can explain train vs test behavior for dropout
- You can list alternative stochastic regularizers and their intuition
- You can propose reasonable p-values and where to apply dropout
