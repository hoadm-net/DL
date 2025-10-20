# Concept 4: Gradient Descent

## 🎯 Objective
Understand how to use gradients to optimize neural network parameters and minimize loss functions.

---

## 📖 Detailed theory

### **What is Gradient Descent?**

Gradient Descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function. It uses the gradients (computed via backpropagation) to determine the direction and magnitude of parameter updates.

### **Why do we need Gradient Descent?**

1. **Optimization:** Find parameters that minimize loss
2. **Learning:** Enable the model to improve from data
3. **Generalization:** Core algorithm for all neural network training

### **The Optimization Intuition**

Think of loss as a mountainous landscape:
- **Current position:** Current parameter values
- **Gradient:** Steepest uphill direction
- **Goal:** Move downhill (negative gradient) to reach valley (minimum loss)

---

## 🧮 Detailed mathematics

### **Basic Update Rule**

For any parameter $\theta$:
$$\theta^{(t+1)} = \theta^{(t)} - \alpha \frac{\partial L}{\partial \theta}$$

Where:
- $\theta^{(t)}$: Parameter at iteration $t$
- $\alpha$: Learning rate (step size)
- $\frac{\partial L}{\partial \theta}$: Gradient of loss w.r.t. parameter

### **Linear Layer Updates**

For our linear model $y = Wx + b$:

**Weight update:**
$$W^{(t+1)} = W^{(t)} - \alpha \frac{\partial L}{\partial W}$$

**Bias update:**
$$b^{(t+1)} = b^{(t)} - \alpha \frac{\partial L}{\partial b}$$

### **Learning Rate Impact**

The learning rate $\alpha$ controls the step size:

**Too small ($\alpha \ll 1$):**
- Slow convergence
- May get stuck in local minima
- Safe but inefficient

**Too large ($\alpha \gg 1$):**
- May overshoot minimum
- Training becomes unstable
- Loss can diverge

**Just right ($\alpha \approx 0.01-0.1$):**
- Smooth convergence
- Reasonable training speed
- Good balance of speed vs stability

### **Convergence Analysis**

**Convex functions (like MSE):**
- Guaranteed to reach global minimum
- Convergence rate depends on learning rate
- No local minima to worry about

**Non-convex functions (deep networks):**
- May reach local minima
- Good local minima often sufficient
- Advanced techniques help escape bad minima

---

## 💻 Exercise 1.4: Gradient descent implementation

### Problem description
Implement gradient descent optimization for a simple linear regression problem.

### Requirements
1. **Function 1:** `gradient_descent_step(params, gradients, learning_rate)`
2. **Function 2:** `train_linear_model(X, y, epochs, learning_rate)`
3. **Function 3:** `learning_rate_experiment()`
4. **Visualization:** Loss curves and parameter evolution

---

### Function 1: Single gradient descent step

```python
def gradient_descent_step(params, gradients, learning_rate):
    """
    Perform one step of gradient descent
    
    Args:
        params (dict): Current parameters {'W': W, 'b': b}
        gradients (dict): Computed gradients {'grad_W': grad_W, 'grad_b': grad_b}  
        learning_rate (float): Step size for updates
        
    Returns:
        new_params (dict): Updated parameters
        
    Formula:
        new_param = old_param - learning_rate * gradient
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
import numpy as np

# Test 1: Simple step
params = {'W': np.array([[1.0]]), 'b': np.array([0.5])}
gradients = {'grad_W': np.array([[0.2]]), 'grad_b': np.array([0.1])}
learning_rate = 0.1

new_params = gradient_descent_step(params, gradients, learning_rate)

# Expected: W = 1.0 - 0.1*0.2 = 0.98, b = 0.5 - 0.1*0.1 = 0.49
assert np.allclose(new_params['W'], [[0.98]])
assert np.allclose(new_params['b'], [0.49])

# Test 2: Large gradients
params = {'W': np.array([[2.0]]), 'b': np.array([1.0])}
gradients = {'grad_W': np.array([[5.0]]), 'grad_b': np.array([3.0])}
learning_rate = 0.01

new_params = gradient_descent_step(params, gradients, learning_rate)

# Expected: W = 2.0 - 0.01*5.0 = 1.95, b = 1.0 - 0.01*3.0 = 0.97
assert np.allclose(new_params['W'], [[1.95]])
assert np.allclose(new_params['b'], [0.97])

# Test 3: Multi-dimensional
params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]]), 'b': np.array([0.1, 0.2])}
gradients = {'grad_W': np.array([[0.1, 0.2], [0.3, 0.4]]), 'grad_b': np.array([0.05, 0.1])}
learning_rate = 0.5

new_params = gradient_descent_step(params, gradients, learning_rate)

expected_W = np.array([[0.95, 1.9], [2.85, 3.8]])  # Original - 0.5 * gradients
expected_b = np.array([0.075, 0.15])

assert np.allclose(new_params['W'], expected_W)
assert np.allclose(new_params['b'], expected_b)
```

---

### Function 2: Complete training loop

```python
def train_linear_model(X, y, epochs=100, learning_rate=0.01, verbose=True):
    """
    Train a linear model using gradient descent
    
    Args:
        X (numpy.ndarray): Input features, shape (m, n)
        y (numpy.ndarray): Target values, shape (m, 1)
        epochs (int): Number of training iterations
        learning_rate (float): Step size for gradient descent
        verbose (bool): Whether to print training progress
        
    Returns:
        params (dict): Final trained parameters
        history (dict): Training history (losses, params over time)
        
    Training process:
        1. Initialize parameters randomly
        2. For each epoch:
           a. Forward pass: compute predictions
           b. Compute loss (MSE)
           c. Backward pass: compute gradients  
           d. Update parameters using gradient descent
           e. Record loss and parameters
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Perfect linear data
np.random.seed(42)
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = 2 * X + 1  # True relationship: y = 2x + 1

params, history = train_linear_model(X, y, epochs=1000, learning_rate=0.01)

# Should learn W ≈ 2, b ≈ 1
assert abs(params['W'][0, 0] - 2.0) < 0.1
assert abs(params['b'][0] - 1.0) < 0.1

# Loss should decrease over time
assert history['losses'][-1] < history['losses'][0]
assert history['losses'][-1] < 0.01  # Should achieve low loss

# Test 2: Noisy data
np.random.seed(123)
X = np.random.randn(50, 1)
y = 1.5 * X + 0.5 + 0.1 * np.random.randn(50, 1)  # y = 1.5x + 0.5 + noise

params, history = train_linear_model(X, y, epochs=500, learning_rate=0.1)

# Should approximately learn W ≈ 1.5, b ≈ 0.5 (within noise tolerance)
assert abs(params['W'][0, 0] - 1.5) < 0.2
assert abs(params['b'][0] - 0.5) < 0.2
```

---

### Function 3: Learning rate experiments

```python
def learning_rate_experiment(X, y, learning_rates=[0.001, 0.01, 0.1, 1.0], epochs=200):
    """
    Compare different learning rates on same dataset
    
    Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target data
        learning_rates (list): List of learning rates to test
        epochs (int): Training epochs for each experiment
        
    Returns:
        results (dict): Results for each learning rate
        
    For each learning rate:
        1. Train model independently
        2. Record loss curves
        3. Record final parameters
        4. Record training stability
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test learning rate effects
np.random.seed(42)
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([[3], [5], [7], [9]], dtype=float)  # y = 2x + 1

results = learning_rate_experiment(X, y, learning_rates=[0.001, 0.1, 0.5], epochs=100)

# Small learning rate should converge slowly but steadily
small_lr_losses = results[0.001]['losses']
assert small_lr_losses[0] > small_lr_losses[-1]  # Should decrease
assert small_lr_losses[-1] > small_lr_losses[-10]  # But still decreasing at end

# Medium learning rate should converge faster
medium_lr_losses = results[0.1]['losses']
assert medium_lr_losses[-10] < small_lr_losses[-10]  # Faster convergence

# Large learning rate might be unstable
large_lr_losses = results[0.5]['losses']
# Check if loss exploded (gradient of losses should be large if unstable)
loss_gradient = np.gradient(large_lr_losses)
max_gradient = np.max(np.abs(loss_gradient))
# This test checks for instability patterns
```

---

### Function 4: Visualization

```python
def visualize_training_process(X, y, learning_rates=[0.01, 0.1, 0.5]):
    """
    Create comprehensive visualizations of training process
    
    Plots to create:
    1. Loss curves for different learning rates
    2. Parameter evolution over training
    3. Decision boundary evolution (for 1D case)
    4. Learning rate sensitivity analysis
    """
    # YOUR CODE HERE
    pass
```

**Expected visualizations:**

1. **Loss Curves Plot:**
   - X-axis: Epochs
   - Y-axis: Loss (log scale)
   - Multiple curves for different learning rates
   - Show convergence patterns

2. **Parameter Evolution:**
   - Track how W and b change over training
   - Show path in parameter space
   - Visualize convergence to optimal values

3. **Prediction Evolution:**
   - Show how model predictions improve
   - Animate regression line fitting data
   - Compare initial vs final predictions

---

## 🔍 Key insights

### **1. Learning Rate is Critical**
- **Too small:** Slow but steady progress
- **Too large:** Fast but potentially unstable
- **Adaptive methods:** Adjust learning rate during training

### **2. Convergence Patterns**
- **Smooth decrease:** Good learning rate choice
- **Oscillations:** Learning rate might be too large
- **Plateau:** May have reached minimum or need different rate

### **3. Loss Landscape Understanding**
- **Convex problems:** Single global minimum, predictable convergence
- **Non-convex problems:** Multiple local minima, more complex dynamics
- **Saddle points:** Gradients are zero but not at minimum

---

## ⚡ Advanced topics (optional)

### **Learning Rate Scheduling**
```python
# Exponential decay
lr_t = lr_0 * decay_rate^(epoch / decay_steps)

# Step decay  
lr_t = lr_0 * drop_factor^floor(epoch / epochs_drop)

# Cosine annealing
lr_t = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / max_epochs)) / 2
```

### **Momentum**
```python
# Add momentum to gradient descent
v_t = momentum * v_{t-1} + learning_rate * gradient
param_t = param_{t-1} - v_t
```

---

## ✅ Success criteria

After completing this exercise, you should:

1. **✅ Understand optimization:** How gradients guide parameter updates
2. **✅ Implement correctly:** All functions pass test cases
3. **✅ Choose learning rates:** Know how to set and tune learning rates
4. **✅ Diagnose problems:** Identify convergence issues from loss curves
5. **✅ Visualize training:** Create informative plots of training process

---

## 🔗 Next steps

After mastering gradient descent:
- **Exercise 1.5:** Complete Training Loop - Integrate all concepts
- **Lab 02:** Activation Functions - Add non-linearity to networks

---

## 💡 Pro tips

1. **Start with small learning rates:** Better to converge slowly than diverge
2. **Plot everything:** Loss curves reveal training dynamics
3. **Use adaptive methods:** Adam, RMSprop adjust learning rates automatically
4. **Monitor gradients:** Check for vanishing/exploding gradient problems
5. **Experiment systematically:** Keep detailed logs of hyperparameter experiments

Gradient descent is the engine of deep learning - understand it deeply! 🚀