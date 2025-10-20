# Concept: Learning Rate Schedules and Practice

## 🎯 Objective
Use learning rate schedules and practical training heuristics to improve convergence and generalization.

---

## 📖 LR schedules
- Step decay: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$
- Exponential decay: $\eta_t = \eta_0 \cdot e^{-kt}$
- Cosine annealing: $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$
- Warmup: start with small $\eta$ for a few steps then ramp up
- Reduce-on-plateau: lower $\eta$ when validation metric stops improving

---

## 🧰 Practical tips
- Start with Adam (or SGD+Momentum) and a modest LR (e.g., 1e-3 for Adam)
- Tune batch size and LR together (larger batches often need larger LR)
- Monitor training/validation loss; watch for divergence or overfitting
- Use gradient clipping for unstable sequences (e.g., RNNs)
- Log metrics and use early stopping for robust training

---

## ✅ Success criteria
- You can describe common LR schedules and their formulas
- You can choose a reasonable starting optimizer and LR
- You can articulate a basic training loop’s monitoring/adjustment strategy
