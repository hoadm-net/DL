# Concept 1: Forward Propagation

## 🎯 Objective
Understand how neural networks compute outputs from inputs through linear transformations.

---

## 📖 Detailed theory

### What is forward propagation?

Forward propagation is the process of computing outputs from inputs by passing data forward through the layers of a neural network. This is the first step in the training process.

### Why do we need forward propagation?

1. Make predictions: to see what the current model produces
2. Compare with ground truth: to compute the loss
3. Prepare for the backward pass: outputs are needed to compute gradients

---

## 🧮 Detailed mathematics

### Case 1: Single input, single output

Simplest case — 1 neuron, 1 input:

$$y = wx + b$$

Concrete example:
- Input: $x = 3$
- Weight: $w = 2$ 
- Bias: $b = 1$
- Output: $y = 2 \times 3 + 1 = 7$

### Case 2: Multiple inputs, single output

One neuron, multiple inputs:

$$y = w_1x_1 + w_2x_2 + w_3x_3 + \dots + w_nx_n + b$$

Vector notation:
$$y = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b$$

Concrete example:
- Input: $\mathbf{x} = [1, 2, 3]$
- Weights: $\mathbf{w} = [0.5, -0.3, 0.8]$
- Bias: $b = 0.1$
- Output: $y = 0.5 \times 1 + (-0.3) \times 2 + 0.8 \times 3 + 0.1 = 0.5 - 0.6 + 2.4 + 0.1 = 2.4$

### Case 3: Batch processing (matrix form)

Multiple samples at once:

$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$$

Dimensions:
- $\mathbf{X} \in \mathbb{R}^{m \times n}$: $m$ samples, $n$ features each
- $\mathbf{W} \in \mathbb{R}^{n \times d}$: weights, $d$ output dimensions
- $\mathbf{b} \in \mathbb{R}^{d}$: bias vector
- $\mathbf{Y} \in \mathbb{R}^{m \times d}$: $m$ predictions, each with $d$ dimensions

Concrete example:
```
X = [[1, 2],     W = [[0.5],     b = [0.1]
    [3, 4],          [0.3]]
    [5, 6]]
     
Y = X @ W + b = [[1*0.5 + 2*0.3] + [0.1]] = [[1.2]
              [3*0.5 + 4*0.3]            [2.8]
              [5*0.5 + 6*0.3]]           [4.4]]
```

---

## 🔍 Intuition and meaning

### Why multiplication and addition?

1. Linear combination: each feature contributes proportionally (via its weight)
2. Feature importance: larger weights indicate more influence
3. Offset: the bias shifts the output up or down
4. Simplicity: linear operations are efficient to compute and optimize

### Geometric interpretation

In 2D space:
- $y = wx + b$ is a straight line
- $w$: the slope
- $b$: the y-intercept

In higher dimensions:
- $y = w_1x_1 + w_2x_2 + b$ represents a plane
- $\mathbf{w}$: the normal vector to the plane

---

## 💻 Exercise 1.1: Forward pass implementation

### Problem description
Implement forward propagation for a linear layer from simple to batch cases.

### Requirements
1. Function 1: `linear_forward_single(x, w, b)`
2. Function 2: `linear_forward_vector(x, w, b)`
3. Function 3: `linear_forward_batch(X, W, b)`
4. Testing: comprehensive test cases
5. Visualization: simple plots of linear transformations

---

### Function 1: Single input, single output

```python
def linear_forward_single(x, w, b):
    """
    Forward pass for single input, single output
    
    Args:
        x (float): Input value
        w (float): Weight parameter
        b (float): Bias parameter
        
    Returns:
        y (float): Output value
        
    Formula:
        y = w * x + b
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Basic
assert linear_forward_single(3, 2, 1) == 7  # 2*3 + 1 = 7

# Test 2: Zero weight
assert linear_forward_single(5, 0, 2) == 2  # 0*5 + 2 = 2

# Test 3: Zero bias
assert linear_forward_single(4, 1.5, 0) == 6.0  # 1.5*4 + 0 = 6

# Test 4: Negative values
assert linear_forward_single(-2, 3, -1) == -7  # 3*(-2) + (-1) = -7
```

---

### Function 2: Multiple inputs, single output

```python
def linear_forward_vector(x, w, b):
    """
    Forward pass for multiple inputs, single output
    
    Args:
        x (numpy.ndarray): Input vector of shape (n,)
        w (numpy.ndarray): Weight vector of shape (n,)  
        b (float): Bias scalar
        
    Returns:
        y (float): Output scalar
        
    Formula:
        y = w^T @ x + b = sum(w_i * x_i) + b
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
import numpy as np

# Test 1: Simple case
x = np.array([1, 2, 3])
w = np.array([0.5, -0.3, 0.8])
b = 0.1
expected = 0.5*1 + (-0.3)*2 + 0.8*3 + 0.1  # = 2.4
assert abs(linear_forward_vector(x, w, b) - expected) < 1e-6

# Test 2: All ones
x = np.array([1, 1, 1, 1])
w = np.array([2, 2, 2, 2])
b = 5
expected = 2*4 + 5  # = 13
assert linear_forward_vector(x, w, b) == expected

# Test 3: Orthogonal vectors
x = np.array([1, 0, -1])
w = np.array([0, 1, 0])
b = 0
expected = 0  # dot product = 0
assert linear_forward_vector(x, w, b) == expected
```

---

### Function 3: Batch processing

```python
def linear_forward_batch(X, W, b):
    """
    Forward pass for batch of inputs
    
    Args:
        X (numpy.ndarray): Input batch of shape (m, n)
                          m = batch size, n = input features
        W (numpy.ndarray): Weight matrix of shape (n, d)  
                          d = output dimensions
        b (numpy.ndarray): Bias vector of shape (d,)
        
    Returns:
        Y (numpy.ndarray): Output batch of shape (m, d)
        
    Formula:
        Y = X @ W + b
    """
    # YOUR CODE HERE
    pass
```

**Test Cases:**
```python
# Test 1: Basic batch
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # shape: (3, 2)
W = np.array([[0.5],
              [0.3]])  # shape: (2, 1)
b = np.array([0.1])    # shape: (1,)

expected = np.array([[1.2],   # 1*0.5 + 2*0.3 + 0.1
                     [2.8],   # 3*0.5 + 4*0.3 + 0.1  
                     [4.4]])  # 5*0.5 + 6*0.3 + 0.1
result = linear_forward_batch(X, W, b)
assert np.allclose(result, expected)

# Test 2: Multiple outputs
X = np.array([[1, 2]])  # shape: (1, 2)
W = np.array([[0.5, -0.2],
              [0.3,  0.4]])  # shape: (2, 2)
b = np.array([0.1, -0.1])    # shape: (2,)

expected = np.array([[1.2, 0.5]])  # [1*0.5+2*0.3+0.1, 1*(-0.2)+2*0.4+(-0.1)]
result = linear_forward_batch(X, W, b)
assert np.allclose(result, expected)
```

---

### Function 4: Visualization

```python
def visualize_linear_transformation():
    """
    Visualize how different w and b affect linear transformation
    
    Create plots showing:
    1. Effect of weight (slope)
    2. Effect of bias (y-intercept)
    3. Multiple linear functions
    """
    # YOUR CODE HERE
    # Use matplotlib to create visualizations
    pass
```

**Expected plots:**
1. **Weight effect:** Same bias, different weights
2. **Bias effect:** Same weight, different biases
3. **Combined effect:** Different weight-bias combinations

---

## ✅ Success criteria

After completing Exercise 1.1, you should:

1. Understand the formula $y = wx + b$ and its vector/matrix forms
2. Implement the three functions so all test cases pass
3. Use NumPy vectorization effectively
4. Explain the geometric meaning of linear transformations
5. Debug your implementation with clear, simple tests

---

## 🔗 Next steps

After mastering forward propagation:
- Concept 2: Loss Functions — measure prediction quality
- Exercise 1.2: implement the MSE loss function

---

## 💡 Tips

1. Start simple: implement the single-input case first
2. Test early: verify against hand calculations
3. Use NumPy: leverage vectorized operations
4. Visualize: simple plots help build intuition
5. Debug systematically: print and check intermediate values

Happy coding! 🚀