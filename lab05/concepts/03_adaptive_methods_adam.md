# Concept: Adaptive Methods (Adam)

## 🎯 Objective
Understand adaptive optimization (AdaGrad, RMSProp, Adam) and their mechanics and trade-offs.

---

## 📖 AdaGrad
Accumulates squared gradients per-parameter:
$$\begin{aligned}
\mathbf{G}_t &= \mathbf{G}_{t-1} + (\nabla_\theta \mathcal{L}(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \eta \, \frac{\nabla_\theta \mathcal{L}(\theta_t)}{\sqrt{\mathbf{G}_t} + \epsilon}
\end{aligned}$$
Good for sparse features; decays learning rate over time.

---

## 📖 RMSProp
Exponential moving average of squared gradients:
$$\begin{aligned}
\mathbf{s}_t &= \rho \, \mathbf{s}_{t-1} + (1-\rho) (\nabla_\theta \mathcal{L}(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \eta \, \frac{\nabla_\theta \mathcal{L}(\theta_t)}{\sqrt{\mathbf{s}_t} + \epsilon}
\end{aligned}$$
Stabilizes AdaGrad’s aggressive decay.

---

## 📖 Adam
Combines momentum (first moment) and RMSProp (second moment) with bias correction:
$$\begin{aligned}
\mathbf{m}_t &= \beta_1 \, \mathbf{m}_{t-1} + (1-\beta_1) \, \nabla_\theta \mathcal{L}(\theta_t) \\
\mathbf{v}_t &= \beta_2 \, \mathbf{v}_{t-1} + (1-\beta_2) \, (\nabla_\theta \mathcal{L}(\theta_t))^2 \\
\hat{\mathbf{m}}_t &= \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
\end{aligned}$$

Defaults: $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$.

---

## 🧠 Intuition and trade-offs
- AdaGrad excels on sparse problems; may stall later due to decay
- RMSProp moderates AdaGrad’s decay; works well in practice
- Adam is a strong default; sometimes plateaus or generalizes worse than SGD+Momentum

---

## ✅ Success criteria
- You can write the update rules for AdaGrad/RMSProp/Adam
- You can explain the role of moving averages and bias correction
- You can justify when to prefer Adam vs SGD+Momentum
