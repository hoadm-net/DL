# Concept: SGD and Mini-batch Training

## 🎯 Objective
Understand stochastic and mini-batch gradient descent, variance vs. efficiency, and practical batch sizing.

---

## 📖 Definitions
- Full-batch GD: updates using all samples per step
- Stochastic GD (SGD): updates using a single sample per step
- Mini-batch SGD: updates using a subset (batch) of samples per step

Update rule (mini-batch, mean loss):
$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \, \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}(\theta; x_i, y_i)$$

---

## 🧠 Intuition
- Mini-batches reduce gradient variance vs pure SGD and improve efficiency vs full batch
- Batch size trades off stability (larger) vs exploration/noise (smaller)
- Typical sizes: 32–512; depends on hardware and problem

---

## ⚖️ Practical notes
- Shuffle data each epoch
- Use mean loss across a mini-batch
- Keep batch sizes aligned with hardware (e.g., powers of 2 for GPUs)

---

## ✅ Success criteria
- You can define GD, SGD, and mini-batch SGD
- You can write and explain the mini-batch update formula
- You can reason about batch size effects on convergence
