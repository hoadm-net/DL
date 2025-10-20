# Concept 3: Backward Propagation

## 🎯 Objective
Understand how to compute gradients using the chain rule to optimize neural network parameters.

---

## 📖 Detailed theory

### **What is Backward Propagation?**

Backward Propagation (backprop) is the process of computing gradients of the loss function with respect to model parameters. It uses the **chain rule** from calculus to efficiently compute how much each parameter contributes to the overall error.

### **Why do we need Backward Propagation?**

1. **Optimization:** Need gradients to update parameters
2. **Efficiency:** Computes all gradients in one backward pass
3. **Foundation:** Core algorithm that makes deep learning possible

### **The Chain Rule Intuition**

If we have a composition of functions: $z = f(g(x))$, then:
$$\frac{dz}{dx} = \frac{dz}{dg} \cdot \frac{dg}{dx}$$

In neural networks, this becomes:
$$\frac{\partial \text{Loss}}{\partial \text{parameter}} = \frac{\partial \text{Loss}}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{parameter}}$$

---

## 🧮 Detailed mathematics

### **Simple Linear Case**

Given:
- Forward: $y = wx + b$
- Loss: $L = \frac{1}{2}(y - \hat{y})^2$

**Step 1: Gradient w.r.t. output**
$$\frac{\partial L}{\partial y} = y - \hat{y}$$

**Step 2: Gradient w.r.t. weight**
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} = (y - \hat{y}) \cdot x$$

**Step 3: Gradient w.r.t. bias**
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial b} = (y - \hat{y}) \cdot 1 = y - \hat{y}$$

### **Batch Processing**

For batch size $m$:

**Gradient w.r.t. weights:**
$$\frac{\partial L}{\partial W} = \frac{1}{m} X^T (Y_{\text{pred}} - Y_{\text{true}})$$

**Gradient w.r.t. bias:**
$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})$$

### **Computational Graph**

```
Input (x) → [Weight (w)] → Linear (wx+b) → Output (y) → Loss (L)
              ↑                ↑              ↑         ↑
         grad_w ←─────── grad_linear ←─── grad_y ←─ grad_L
```

---

## 💻 Exercise 1.3: Backward pass implementation

### Problem description
Implement backward propagation for a simple linear layer with MSE loss.

### Requirements
1. **Function 1:** `compute_gradients_single(x, y_pred, y_true, w, b)`
2. **Function 2:** `compute_gradients_batch(X, Y_pred, Y_true, W, b)`
3. **Function 3:** `numerical_gradient_check()`
4. **Visualization:** Gradient flow diagram

---

### Function 1: Single sample gradients

```python
def compute_gradients_single(x, y_pred, y_true, w, b):
    """
    Compute gradients for single training example
    
    Args:
        x (float): Input value
        y_pred (float): Predicted output  
        y_true (float): Ground truth
        w (float): Current weight
        b (float): Current bias
        
    Returns:
        grad_w (float): Gradient w.r.t. weight
        grad_b (float): Gradient w.r.t. bias
        
    Formulas:
        grad_w = (y_pred - y_true) * x
        grad_b = (y_pred - y_true)
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Simple case
x, y_pred, y_true, w, b = 2.0, 5.0, 3.0, 1.0, 0.0
grad_w, grad_b = compute_gradients_single(x, y_pred, y_true, w, b)
assert grad_w == 4.0  # (5-3) * 2 = 4
assert grad_b == 2.0  # (5-3) = 2

# Test 2: Zero error
x, y_pred, y_true, w, b = 1.0, 2.0, 2.0, 1.0, 1.0
grad_w, grad_b = compute_gradients_single(x, y_pred, y_true, w, b)
assert grad_w == 0.0  # Perfect prediction
assert grad_b == 0.0

# Test 3: Negative gradients
x, y_pred, y_true, w, b = 3.0, 1.0, 4.0, 2.0, -1.0
grad_w, grad_b = compute_gradients_single(x, y_pred, y_true, w, b)
assert grad_w == -9.0  # (1-4) * 3 = -9
assert grad_b == -3.0  # (1-4) = -3
```

---

### Function 2: Batch gradients

```python
def compute_gradients_batch(X, Y_pred, Y_true, W, b):
    """
    Compute gradients for batch of training examples
    
    Args:
        X (numpy.ndarray): Input batch, shape (m, n)
        Y_pred (numpy.ndarray): Predictions, shape (m, d)  
        Y_true (numpy.ndarray): Ground truth, shape (m, d)
        W (numpy.ndarray): Weights, shape (n, d)
        b (numpy.ndarray): Bias, shape (d,)
        
    Returns:
        grad_W (numpy.ndarray): Gradient w.r.t. weights, shape (n, d)
        grad_b (numpy.ndarray): Gradient w.r.t. bias, shape (d,)
        
    Formulas:
        grad_W = (1/m) * X.T @ (Y_pred - Y_true)
        grad_b = (1/m) * sum(Y_pred - Y_true)
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
import numpy as np

# Test 1: Simple batch
X = np.array([[1.0], [2.0], [3.0]])  # (3, 1)
Y_pred = np.array([[2.0], [4.0], [6.0]])  # (3, 1)  
Y_true = np.array([[1.0], [3.0], [5.0]])  # (3, 1)
W = np.array([[1.0]])  # (1, 1)
b = np.array([0.0])    # (1,)

grad_W, grad_b = compute_gradients_batch(X, Y_pred, Y_true, W, b)

# Expected: grad_W = (1/3) * [1,2,3].T @ [1,1,1] = (1/3) * 6 = 2.0
# Expected: grad_b = (1/3) * sum([1,1,1]) = 1.0
assert np.allclose(grad_W, [[2.0]])
assert np.allclose(grad_b, [1.0])

# Test 2: Multi-dimensional
X = np.array([[1.0, 2.0]])  # (1, 2)
Y_pred = np.array([[3.0, 4.0]])  # (1, 2)
Y_true = np.array([[2.0, 3.0]])  # (1, 2)  
W = np.array([[1.0, 0.5], [0.0, 1.0]])  # (2, 2)
b = np.array([0.0, 0.0])  # (2,)

grad_W, grad_b = compute_gradients_batch(X, Y_pred, Y_true, W, b)
# Error = [1.0, 1.0]
# grad_W = X.T @ error = [[1.0], [2.0]] @ [[1.0, 1.0]] = [[1.0, 1.0], [2.0, 2.0]]
expected_grad_W = np.array([[1.0, 1.0], [2.0, 2.0]])
expected_grad_b = np.array([1.0, 1.0])

assert np.allclose(grad_W, expected_grad_W)
assert np.allclose(grad_b, expected_grad_b)
```

---

### Function 3: Numerical gradient checking

```python
def numerical_gradient_check(func, x, epsilon=1e-7):
    """
    Verify analytical gradients using numerical approximation
    
    Args:
        func: Function that returns (loss, analytical_grad)
        x: Point to check gradients at
        epsilon: Small perturbation for numerical gradient
        
    Returns:
        bool: True if gradients match within tolerance
        
    Formula:
        numerical_grad = (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test function: f(x) = x^2, df/dx = 2x
def test_function(x):
    loss = x**2
    analytical_grad = 2*x
    return loss, analytical_grad

# Check at x = 3
is_correct = numerical_gradient_check(test_function, 3.0)
assert is_correct == True

# Check at x = 0
is_correct = numerical_gradient_check(test_function, 0.0)
assert is_correct == True
```

---

### Function 4: Gradient visualization

```python
def visualize_gradient_flow():
    """
    Visualize how gradients flow backward through computation graph
    
    Create plots showing:
    1. Forward pass computation
    2. Backward pass gradient flow  
    3. Gradient magnitudes at each step
    """
    # YOUR CODE HERE
    pass
```

---

## 🔍 Key insights

### **1. Chain Rule Magic**
- Gradients flow backward through computational graph
- Each operation contributes its local gradient
- Product of local gradients gives total gradient

### **2. Efficiency**
- One backward pass computes all gradients
- Reuses computations from forward pass
- Scales to arbitrarily deep networks

### **3. Numerical Stability**
- Always check analytical gradients numerically
- Small errors can compound in deep networks
- Proper gradient scaling prevents vanishing/exploding

---

## ✅ Success criteria

After completing this exercise, you should:

1. **✅ Understand chain rule:** How gradients compose through operations
2. **✅ Implement correctly:** All test cases pass
3. **✅ Verify numerically:** Analytical gradients match numerical approximation
4. **✅ Visualize flow:** Understand gradient propagation graphically
5. **✅ Debug systematically:** Can identify and fix gradient computation errors

---

## 🔗 Next steps

After mastering backward propagation:
- **Concept 4:** Gradient Descent - Use gradients to optimize parameters
- **Exercise 1.4:** Implement parameter updates and training loops

---

## 💡 Pro tips

1. **Always verify:** Use numerical gradient checking during development
2. **Think locally:** Each operation only needs its local gradient
3. **Vectorize everything:** Batch operations are much more efficient
4. **Visualize graphs:** Draw computation graphs to understand flow
5. **Start simple:** Master linear case before moving to complex architectures

Backward propagation is the heart of deep learning - master it well! 🎯