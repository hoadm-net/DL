# Concept: Momentum and Nesterov

## 🎯 Objective
Accelerate convergence and reduce oscillations with momentum-based gradient methods.

---

## 📖 Momentum
Velocity update and parameter update:
$$\begin{aligned}
\mathbf{v}_{t+1} &= \mu \, \mathbf{v}_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t + \mathbf{v}_{t+1}
\end{aligned}$$

- $\mu$ (0.8–0.99): momentum coefficient
- $\eta$: learning rate

Intuition: exponential moving average of past gradients; damps oscillations and builds speed along consistent directions.

---

## 📖 Nesterov Accelerated Gradient (NAG)
Lookahead gradient at the approximate future position:
$$\begin{aligned}
\tilde{\theta}_t &= \theta_t + \mu \, \mathbf{v}_t \\
\mathbf{v}_{t+1} &= \mu \, \mathbf{v}_t - \eta \, \nabla_\theta \mathcal{L}(\tilde{\theta}_t) \\
\theta_{t+1} &= \theta_t + \mathbf{v}_{t+1}
\end{aligned}$$

Intuition: corrects the direction using a lookahead; often more responsive than classical momentum.

---

## ⚖️ Practical notes
- Start with momentum 0.9; tune if needed
- Combine with LR schedules for best results
- Sensitive to scaling; consider normalization and good initialization

---

## ✅ Success criteria
- You can write momentum and NAG update rules
- You can explain why momentum reduces zig-zagging
- You can justify when NAG may outperform classical momentum
