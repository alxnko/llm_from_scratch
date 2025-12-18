"""
Pure Python math utilities for neural network operations.
No external dependencies - all operations implemented from scratch.
"""

import math
import time


class Random:
    """Simple random number generator using Linear Congruential Generator."""
    
    def __init__(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        self.state = seed
        # LCG parameters (same as glibc)
        self.a = 1103515245
        self.c = 12345
        self.m = 2**31
    
    def _next(self):
        """Generate next random integer."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self):
        """Return random float between 0 and 1."""
        return self._next() / self.m
    
    def uniform(self, low, high):
        """Return random float between low and high."""
        return low + self.random() * (high - low)
    
    def gauss(self, mu=0.0, sigma=1.0):
        """Generate random number from Gaussian distribution using Box-Muller."""
        u1 = self.random()
        u2 = self.random()
        # Avoid log(0)
        while u1 == 0:
            u1 = self.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + z0 * sigma
    
    def choice(self, items):
        """Randomly select an item from a list."""
        idx = int(self.random() * len(items))
        return items[min(idx, len(items) - 1)]
    
    def shuffle(self, items):
        """Shuffle a list in place."""
        n = len(items)
        for i in range(n - 1, 0, -1):
            j = int(self.random() * (i + 1))
            items[i], items[j] = items[j], items[i]
        return items


# Global random instance
_random = Random()


class Matrix:
    """Matrix operations implemented in pure Python."""
    
    @staticmethod
    def zeros(rows, cols):
        """Create a matrix filled with zeros."""
        return [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def ones(rows, cols):
        """Create a matrix filled with ones."""
        return [[1.0 for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def random_init(rows, cols, scale=0.1, rng=None):
        """Create a matrix with random values (Xavier-like initialization)."""
        if rng is None:
            rng = _random
        # Xavier initialization scale
        std = scale * math.sqrt(2.0 / (rows + cols))
        return [[rng.gauss(0, std) for _ in range(cols)] for _ in range(rows)]
    
    @staticmethod
    def shape(matrix):
        """Get shape of matrix."""
        if not matrix:
            return (0, 0)
        if isinstance(matrix[0], list):
            return (len(matrix), len(matrix[0]))
        return (len(matrix),)
    
    @staticmethod
    def add(a, b):
        """Element-wise addition of two matrices."""
        rows, cols = Matrix.shape(a)
        return [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def subtract(a, b):
        """Element-wise subtraction of two matrices."""
        rows, cols = Matrix.shape(a)
        return [[a[i][j] - b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def multiply(a, b):
        """Matrix multiplication (dot product) - optimized version."""
        rows_a = len(a)
        cols_a = len(a[0]) if a else 0
        rows_b = len(b)
        cols_b = len(b[0]) if b else 0
        
        if cols_a != rows_b:
            raise ValueError(f"Cannot multiply matrices: {cols_a} != {rows_b}")
        
        # Transpose b for cache-friendly access (rows instead of columns)
        b_T = [[b[k][j] for k in range(rows_b)] for j in range(cols_b)]
        
        # Compute result with optimized inner loop
        result = []
        for i in range(rows_a):
            row_a = a[i]
            result_row = []
            for j in range(cols_b):
                col_b = b_T[j]
                # Optimized dot product
                total = sum(row_a[k] * col_b[k] for k in range(cols_a))
                result_row.append(total)
            result.append(result_row)
        return result
    
    @staticmethod
    def elementwise_multiply(a, b):
        """Element-wise multiplication (Hadamard product)."""
        rows, cols = Matrix.shape(a)
        return [[a[i][j] * b[i][j] for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def scalar_multiply(matrix, scalar):
        """Multiply matrix by a scalar."""
        rows, cols = Matrix.shape(matrix)
        return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]
    
    @staticmethod
    def transpose(matrix):
        """Transpose a matrix."""
        rows, cols = Matrix.shape(matrix)
        return [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    
    @staticmethod
    def vector_to_row(vector):
        """Convert 1D vector to row matrix."""
        return [vector]
    
    @staticmethod
    def vector_to_col(vector):
        """Convert 1D vector to column matrix."""
        return [[v] for v in vector]
    
    @staticmethod
    def flatten(matrix):
        """Flatten matrix to 1D list."""
        return [val for row in matrix for val in row]
    
    @staticmethod
    def sum_rows(matrix):
        """Sum along rows, returning column vector."""
        rows, cols = Matrix.shape(matrix)
        return [[sum(matrix[i])] for i in range(rows)]
    
    @staticmethod
    def sum_cols(matrix):
        """Sum along columns, returning row vector."""
        rows, cols = Matrix.shape(matrix)
        return [[sum(matrix[i][j] for i in range(rows)) for j in range(cols)]]
    
    @staticmethod
    def mean(matrix):
        """Calculate mean of all elements."""
        flat = Matrix.flatten(matrix)
        return sum(flat) / len(flat) if flat else 0.0
    
    @staticmethod
    def variance(matrix):
        """Calculate variance of all elements."""
        flat = Matrix.flatten(matrix)
        if not flat:
            return 0.0
        mean = sum(flat) / len(flat)
        return sum((x - mean) ** 2 for x in flat) / len(flat)
    
    @staticmethod
    def clip(matrix, min_val, max_val):
        """Clip values to range."""
        rows, cols = Matrix.shape(matrix)
        return [[max(min_val, min(max_val, matrix[i][j])) for j in range(cols)] for i in range(rows)]


class Activations:
    """Activation functions and their derivatives."""
    
    @staticmethod
    def relu(x):
        """ReLU activation."""
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[max(0, v) for v in row] for row in x]
            return [max(0, v) for v in x]
        return max(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU."""
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[1.0 if v > 0 else 0.0 for v in row] for row in x]
            return [1.0 if v > 0 else 0.0 for v in x]
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def tanh(x):
        """Tanh activation."""
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[math.tanh(v) for v in row] for row in x]
            return [math.tanh(v) for v in x]
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh."""
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[1 - math.tanh(v)**2 for v in row] for row in x]
            return [1 - math.tanh(v)**2 for v in x]
        return 1 - math.tanh(x)**2
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation."""
        def _sigmoid(v):
            # Clip to prevent overflow
            v = max(-500, min(500, v))
            return 1.0 / (1.0 + math.exp(-v))
        
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[_sigmoid(v) for v in row] for row in x]
            return [_sigmoid(v) for v in x]
        return _sigmoid(x)
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid."""
        s = Activations.sigmoid(x)
        if isinstance(s, list):
            if isinstance(s[0], list):
                return [[v * (1 - v) for v in row] for row in s]
            return [v * (1 - v) for v in s]
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        """Softmax activation for 1D or 2D input."""
        if isinstance(x[0], list):
            # 2D: apply softmax to each row
            return [Activations.softmax(row) for row in x]
        
        # 1D softmax
        # Subtract max for numerical stability
        max_val = max(x)
        exp_vals = [math.exp(v - max_val) for v in x]
        sum_exp = sum(exp_vals)
        return [v / sum_exp for v in exp_vals]
    
    @staticmethod
    def gelu(x):
        """GELU activation (approximate)."""
        def _gelu(v):
            return 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
        
        if isinstance(x, list):
            if isinstance(x[0], list):
                return [[_gelu(v) for v in row] for row in x]
            return [_gelu(v) for v in x]
        return _gelu(x)


class Loss:
    """Loss functions for training."""
    
    @staticmethod
    def cross_entropy(predicted, target_idx):
        """
        Cross-entropy loss for classification.
        predicted: softmax probabilities (1D list)
        target_idx: index of correct class
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        prob = max(epsilon, min(1 - epsilon, predicted[target_idx]))
        return -math.log(prob)
    
    @staticmethod
    def cross_entropy_gradient(predicted, target_idx):
        """
        Gradient of cross-entropy loss w.r.t. softmax output.
        Returns gradient vector.
        """
        grad = predicted.copy()
        grad[target_idx] -= 1.0
        return grad
    
    @staticmethod
    def mse(predicted, target):
        """Mean squared error loss."""
        if isinstance(predicted[0], list):
            # 2D
            total = 0
            count = 0
            for i in range(len(predicted)):
                for j in range(len(predicted[0])):
                    total += (predicted[i][j] - target[i][j]) ** 2
                    count += 1
            return total / count
        else:
            # 1D
            return sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)
    
    @staticmethod
    def mse_gradient(predicted, target):
        """Gradient of MSE loss."""
        n = len(predicted) if not isinstance(predicted[0], list) else len(predicted) * len(predicted[0])
        if isinstance(predicted[0], list):
            return [[(2 * (predicted[i][j] - target[i][j]) / n) 
                     for j in range(len(predicted[0]))] 
                    for i in range(len(predicted))]
        return [(2 * (p - t) / n) for p, t in zip(predicted, target)]
