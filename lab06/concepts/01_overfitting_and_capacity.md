# Concept: Overfitting, Capacity, and Bias–Variance

## 🎯 Objective
Diagnose over/underfitting and understand capacity control through the bias–variance lens.

---

## 📖 Core ideas
- Overfitting: low training error, high validation error
- Underfitting: high training and validation error
- Capacity: model’s ability to fit complex patterns (depth, width, parameters)
- Bias–variance trade-off: increasing capacity ↓ bias, ↑ variance

---

## 🧠 Intuition
- Too simple → can’t fit data (high bias)
- Too complex → memorizes noise (high variance)
- Sweet spot → minimal validation error

Learning curves (error vs. training size/epochs) help diagnose which regime you’re in.

---

## 🧰 Tools
- Cross-validation and a proper validation split
- Data augmentation (images, text): enrich diversity without new labels
- Early stopping: halt when validation worsens

---

## ✅ Success criteria
- You can identify over/underfitting from training vs validation curves
- You can propose capacity adjustments and data-centric fixes
- You can relate bias–variance to regularization choices
