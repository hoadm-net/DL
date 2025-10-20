# Concept 2: Loss Functions

## 🎯 Objective
Understand how to measure prediction quality and why loss functions are essential in machine learning.

---

## 📖 Detailed theory

### What is a loss function?

A loss function measures the difference between model predictions and ground truth values. It quantifies how “good” or “bad” the current model is.

### Why do we need loss functions?

1. Quantify performance: convert “good/bad” into numbers
2. Optimization target: provide an objective to minimize
3. Compare models: evaluate different models fairly
4. Training signal: guide how the model should adjust

### Properties of a good loss function

1. Non-negative: $L \geq 0$ (always positive or zero)
2. Zero at perfect prediction: $L = 0$ when prediction is perfect
3. Differentiable: can compute gradients
4. Problem-appropriate: reflects what you care about

---

## 🧮 Mean Squared Error (MSE)

### Mathematical definition

Single prediction:
$$L = (y_{\text{pred}} - y_{\text{true}})^2$$

Batch predictions:
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2$$

Vectorized form:
$$\text{MSE} = \frac{1}{m} \lVert\mathbf{y}_{\text{pred}} - \mathbf{y}_{\text{true}}\rVert^2_2$$

### Why square the error?

1. Always positive: $(\text{error})^2 \geq 0$
2. Penalizes large errors more: bigger mistakes hurt more
3. Smooth gradient: easy to optimize
4. Mathematical convenience: simple derivative

### Concrete examples

Case 1: perfect prediction
- $y_{\text{true}} = 5$, $y_{\text{pred}} = 5$ → $L = 0$

Case 2: small error
- $y_{\text{true}} = 5$, $y_{\text{pred}} = 5.1$ → $L = 0.01$

Case 3: large error
- $y_{\text{true}} = 5$, $y_{\text{pred}} = 7$ → $L = 4$

---

## 🔍 Properties of MSE

1. Sensitivity to outliers

```
Normal errors: [0.1, 0.2, 0.1, 0.15] → MSE = 0.0255
With outlier:  [0.1, 0.2, 0.1, 2.0]  → MSE = 1.0255
```

2. Units
- If targets are in meters, MSE is in meters²

3. Scale dependence
- Target range [0, 1]: MSE = 0.01 might be good
- Target range [0, 1000]: MSE = 0.01 is excellent

4. Convexity
- MSE is convex → unique global minimum (for linear models)

---

## 📊 Comparing with other loss functions

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}|$$

MSE vs MAE:
- MSE: penalizes large errors more, smooth gradient
- MAE: robust to outliers, non-smooth at zero

### Huber loss (robust)
$$L_{\delta}(y, f(x)) = \begin{cases}
\frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \leq \delta \\
\delta |y - f(x)| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

Combines the benefits: MSE for small errors, MAE for large errors.

---

## 💻 Exercise 1.2: Loss function implementation

### Problem description
Implement the MSE loss function and explore its properties.

### Requirements
1. Function 1: `mse_loss_single(y_pred, y_true)`
2. Function 2: `mse_loss_batch(y_pred, y_true)`
3. Function 3: `mae_loss_batch(y_pred, y_true)` (bonus)
4. Analysis: compare different loss behaviors
5. Visualization: plot loss landscapes

---

### Function 1: Single prediction MSE

```python
def mse_loss_single(y_pred, y_true):
    """
    MSE loss for single prediction
    
    Args:
        y_pred (float): Predicted value
        y_true (float): Ground truth value
        
    Returns:
        loss (float): MSE loss value
        
    Formula:
        loss = (y_pred - y_true)^2
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Perfect prediction
assert mse_loss_single(5.0, 5.0) == 0.0

# Test 2: Small error
assert abs(mse_loss_single(5.1, 5.0) - 0.01) < 1e-10

# Test 3: Large error  
assert mse_loss_single(7.0, 5.0) == 4.0

# Test 4: Negative error (same as positive)
assert mse_loss_single(3.0, 5.0) == mse_loss_single(7.0, 5.0)  # Both have |error| = 2

# Test 5: Floating point precision
assert abs(mse_loss_single(1.23456, 1.23455) - 1e-10) < 1e-15
```

---

### Function 2: Batch MSE loss

```python
def mse_loss_batch(y_pred, y_true):
    """
    MSE loss for batch of predictions
    
    Args:
        y_pred (numpy.ndarray): Predicted values, shape (m,) or (m, d)
        y_true (numpy.ndarray): Ground truth values, shape (m,) or (m, d)
        
    Returns:
        loss (float): Average MSE loss across batch
        
    Formula:
        loss = (1/m) * sum((y_pred - y_true)^2)
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
import numpy as np

# Test 1: Perfect predictions
y_pred = np.array([1, 2, 3, 4])
y_true = np.array([1, 2, 3, 4])
assert mse_loss_batch(y_pred, y_true) == 0.0

# Test 2: Uniform errors
y_pred = np.array([1.1, 2.1, 3.1, 4.1])
y_true = np.array([1, 2, 3, 4])
expected = 0.01  # All errors = 0.1, squared = 0.01
assert abs(mse_loss_batch(y_pred, y_true) - expected) < 1e-10

# Test 3: Mixed errors
y_pred = np.array([1, 3, 5])
y_true = np.array([2, 2, 6])
# Errors: [-1, 1, -1], squared: [1, 1, 1], mean: 1.0
expected = 1.0
assert mse_loss_batch(y_pred, y_true) == expected

# Test 4: Multi-dimensional
y_pred = np.array([[1, 2], [3, 4]])
y_true = np.array([[1.1, 2.1], [2.9, 3.9]])
# Errors: [[-0.1, -0.1], [0.1, 0.1]]
# Squared: [[0.01, 0.01], [0.01, 0.01]]
# Mean: 0.01
expected = 0.01
assert abs(mse_loss_batch(y_pred, y_true) - expected) < 1e-10
```

---

### Function 3: MAE loss (bonus)

```python
def mae_loss_batch(y_pred, y_true):
    """
    MAE (Mean Absolute Error) loss for batch
    
    Args:
        y_pred (numpy.ndarray): Predicted values
        y_true (numpy.ndarray): Ground truth values
        
    Returns:
        loss (float): Average MAE loss
        
    Formula:
        loss = (1/m) * sum(|y_pred - y_true|)
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Compare with MSE on same data
y_pred = np.array([1, 3, 5])
y_true = np.array([2, 2, 6])

mse = mse_loss_batch(y_pred, y_true)  # Should be 1.0
mae = mae_loss_batch(y_pred, y_true)  # Should be 1.0 (same in this case)

assert mse == mae == 1.0

# Test 2: Different behavior with outliers
y_pred = np.array([1, 1, 10])  # One large outlier
y_true = np.array([1, 1, 1])

mse = mse_loss_batch(y_pred, y_true)  # (0^2 + 0^2 + 9^2)/3 = 27
mae = mae_loss_batch(y_pred, y_true)  # (0 + 0 + 9)/3 = 3

assert abs(mse - 27.0) < 1e-10
assert abs(mae - 3.0) < 1e-10
print(f"MSE penalizes outlier more: MSE={mse:.1f}, MAE={mae:.1f}")
```

---

### Function 4: Loss landscape visualization

```python
def visualize_loss_landscape():
    """
    Visualize how loss changes with predictions
    
    Create plots showing:
    1. MSE vs prediction error
    2. MSE vs MAE comparison
    3. Effect of outliers
    """
    # YOUR CODE HERE
    # Create informative plots using matplotlib
    pass
```

**Expected plots:**

1. **Loss vs Error plot:**
   - X-axis: prediction error (-5 to 5)
   - Y-axis: loss value
   - Compare MSE vs MAE curves

2. **Outlier sensitivity:**
   - Show how MSE/MAE change when adding outliers

3. **Batch loss analysis:**
   - Distribution of individual losses vs average loss

---

### Function 5: Loss analysis

```python
def analyze_loss_properties():
    """
    Empirical analysis of loss function properties
    
    Analyze:
    1. Sensitivity to outliers
    2. Scale dependence  
    3. Error distribution effects
    """
    # YOUR CODE HERE
    pass
```

**Analysis tasks:**

1. **Outlier sensitivity test:**
```python
# Normal data
normal_errors = np.random.normal(0, 0.1, 100)
# Data with outliers  
outlier_errors = np.concatenate([normal_errors, [5.0, -4.0]])

# Compare MSE values
```

2. **Scale dependence test:**
```python
# Same relative errors, different scales
small_scale = np.array([1.0, 1.1, 0.9])    # target around 1
large_scale = np.array([100, 110, 90])      # target around 100

# Compare MSE values
```

---

## ✅ Success criteria

After completing Exercise 1.2, you should:

1. Understand the MSE formula: $\text{MSE} = \frac{1}{m} \sum (y_{\text{pred}} - y_{\text{true}})^2$
2. Implement correctly: all test cases pass
3. Understand properties: outlier sensitivity, scale dependence
4. Compare losses: trade-offs between MSE and MAE
5. Visualize: plot and interpret loss landscapes

---

## 🔗 Next steps

After mastering loss functions:
- Concept 3: Backward Propagation — compute gradients of the loss
- Exercise 1.3: implement gradient computation

---

## 💡 Key insights

1. Loss = objective: models learn by minimizing loss
2. Choice matters: different losses lead to different behaviors
3. No free lunch: every loss has trade-offs
4. Domain knowledge: choose a loss that matches your problem
5. Regularization: sometimes modify the loss to prevent overfitting

Understanding loss functions is crucial — they drive the entire learning process. 🎯