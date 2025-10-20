# Concept: Weight Initialization

## 🎯 Objective
Choose initial weights to maintain healthy signal/gradient magnitudes across layers.

---

## 📖 Why initialization matters
- Poor scale → activations saturate or explode
- Gradients vanish/explode → training stalls or diverges
- Good initialization keeps variance stable layer-to-layer

---

## 📐 Variance-preserving schemes
Let $n_{\text{in}}$ be fan-in (inputs to a layer) and $n_{\text{out}}$ be fan-out.

### Xavier/Glorot (tanh/sigmoid)
- Aim: keep activations/gradients variance constant
- Gaussian: $\mathcal{N}(0, \frac{2}{n_{\text{in}} + n_{\text{out}}})$
- Uniform: $\mathcal{U}\big[-\sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\tfrac{6}{n_{\text{in}} + n_{\text{out}}}}\big]$

### He/Kaiming (ReLU)
- Accounts for ReLU’s zeroing of negative inputs
- Gaussian: $\mathcal{N}(0, \frac{2}{n_{\text{in}}})$
- Uniform: $\mathcal{U}\big[-\sqrt{\tfrac{6}{n_{\text{in}}}}, \sqrt{\tfrac{6}{n_{\text{in}}}}\big]$

Biases: initialize to zeros or small constants.

---

## 🧠 Intuition and caveats
- Match scheme to activation: Xavier for tanh/sigmoid, He for ReLU
- BatchNorm can reduce sensitivity to initialization but doesn’t remove it
- Very deep nets may still face gradient issues without additional tricks

---

## ✅ Success criteria
- You can pick Xavier vs He based on activation type
- You can state the common Gaussian/uniform forms and parameters
- You can explain the goal: stable variance through depth
