"""
Neural network layers implemented in pure Python.
Includes Embedding, Self-Attention, Feed-Forward, and Layer Normalization.
"""

import math
from .math_utils import Matrix, Activations, Random


class Embedding:
    """
    Word embedding layer that converts token indices to dense vectors.
    """
    
    def __init__(self, vocab_size, embed_dim, rng=None):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Number of unique tokens
            embed_dim: Dimension of embedding vectors
            rng: Random number generator instance
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rng = rng or Random()
        
        # Initialize embedding matrix with small random values
        self.weights = Matrix.random_init(vocab_size, embed_dim, scale=1.0, rng=self.rng)
        
        # Gradient storage
        self.grad_weights = None
    
    def forward(self, token_indices):
        """
        Look up embeddings for token indices.
        
        Args:
            token_indices: List of token indices
            
        Returns:
            List of embedding vectors (2D matrix: seq_len x embed_dim)
        """
        self.last_input = token_indices
        return [self.weights[idx].copy() for idx in token_indices]
    
    def backward(self, grad_output):
        """
        Compute gradients for embedding weights.
        
        Args:
            grad_output: Gradient from next layer (seq_len x embed_dim)
        """
        # Initialize gradient accumulator
        self.grad_weights = Matrix.zeros(self.vocab_size, self.embed_dim)
        
        # Accumulate gradients for each token position
        for pos, idx in enumerate(self.last_input):
            for j in range(self.embed_dim):
                self.grad_weights[idx][j] += grad_output[pos][j]
    
    def update(self, learning_rate):
        """Update weights using gradients."""
        if self.grad_weights is None:
            return
        for i in range(self.vocab_size):
            for j in range(self.embed_dim):
                self.weights[i][j] -= learning_rate * self.grad_weights[i][j]
    
    def get_params(self):
        """Return parameters for saving."""
        return {"weights": self.weights}
    
    def set_params(self, params):
        """Load parameters."""
        self.weights = params["weights"]


class PositionalEncoding:
    """
    Adds positional information to embeddings using sinusoidal encoding.
    """
    
    def __init__(self, max_seq_len, embed_dim):
        """
        Initialize positional encoding.
        
        Args:
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
        """
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Precompute positional encodings
        self.encoding = self._create_encoding()
    
    def _create_encoding(self):
        """Create sinusoidal positional encoding matrix."""
        encoding = Matrix.zeros(self.max_seq_len, self.embed_dim)
        
        for pos in range(self.max_seq_len):
            for i in range(self.embed_dim):
                if i % 2 == 0:
                    # sin for even indices
                    encoding[pos][i] = math.sin(pos / (10000 ** (i / self.embed_dim)))
                else:
                    # cos for odd indices
                    encoding[pos][i] = math.cos(pos / (10000 ** ((i - 1) / self.embed_dim)))
        
        return encoding
    
    def forward(self, embeddings):
        """
        Add positional encoding to embeddings.
        
        Args:
            embeddings: Token embeddings (seq_len x embed_dim)
            
        Returns:
            Embeddings with positional information added
        """
        seq_len = len(embeddings)
        result = []
        for pos in range(seq_len):
            row = [embeddings[pos][j] + self.encoding[pos][j] 
                   for j in range(self.embed_dim)]
            result.append(row)
        return result
    
    def backward(self, grad_output):
        """Positional encoding has no learnable parameters."""
        return grad_output


class SelfAttention:
    """
    Single-head self-attention layer.
    """
    
    def __init__(self, embed_dim, rng=None):
        """
        Initialize self-attention layer.
        
        Args:
            embed_dim: Dimension of input embeddings
            rng: Random number generator
        """
        self.embed_dim = embed_dim
        self.rng = rng or Random()
        
        # Query, Key, Value projection matrices
        self.W_q = Matrix.random_init(embed_dim, embed_dim, scale=1.0, rng=self.rng)
        self.W_k = Matrix.random_init(embed_dim, embed_dim, scale=1.0, rng=self.rng)
        self.W_v = Matrix.random_init(embed_dim, embed_dim, scale=1.0, rng=self.rng)
        
        # Output projection
        self.W_o = Matrix.random_init(embed_dim, embed_dim, scale=1.0, rng=self.rng)
        
        # Gradient storage
        self.grad_W_q = None
        self.grad_W_k = None
        self.grad_W_v = None
        self.grad_W_o = None
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x):
        """
        Compute self-attention.
        
        Args:
            x: Input embeddings (seq_len x embed_dim)
            
        Returns:
            Attention output (seq_len x embed_dim)
        """
        seq_len = len(x)
        
        # Compute Q, K, V projections
        Q = Matrix.multiply(x, self.W_q)  # seq_len x embed_dim
        K = Matrix.multiply(x, self.W_k)
        V = Matrix.multiply(x, self.W_v)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        K_T = Matrix.transpose(K)
        scores = Matrix.multiply(Q, K_T)  # seq_len x seq_len
        
        # Scale by sqrt(embed_dim)
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = Matrix.scalar_multiply(scores, scale)
        
        # Apply softmax to get attention weights
        attention_weights = Activations.softmax(scores)
        
        # Apply attention to values
        attended = Matrix.multiply(attention_weights, V)  # seq_len x embed_dim
        
        # Output projection
        output = Matrix.multiply(attended, self.W_o)
        
        # Cache for backward pass
        self.cache = {
            "x": x,
            "Q": Q,
            "K": K,
            "V": V,
            "scores": scores,
            "attention_weights": attention_weights,
            "attended": attended,
        }
        
        return output
    
    def backward(self, grad_output):
        """
        Compute gradients through self-attention.
        
        Args:
            grad_output: Gradient from next layer (seq_len x embed_dim)
            
        Returns:
            Gradient w.r.t. input (seq_len x embed_dim)
        """
        x = self.cache["x"]
        Q = self.cache["Q"]
        K = self.cache["K"]
        V = self.cache["V"]
        attention_weights = self.cache["attention_weights"]
        attended = self.cache["attended"]
        
        seq_len = len(x)
        
        # Gradient w.r.t. W_o
        attended_T = Matrix.transpose(attended)
        self.grad_W_o = Matrix.multiply(attended_T, grad_output)
        
        # Gradient w.r.t. attended
        W_o_T = Matrix.transpose(self.W_o)
        grad_attended = Matrix.multiply(grad_output, W_o_T)
        
        # Gradient w.r.t. attention_weights and V
        V_T = Matrix.transpose(V)
        grad_attention = Matrix.multiply(grad_attended, V_T)
        
        attention_T = Matrix.transpose(attention_weights)
        grad_V = Matrix.multiply(attention_T, grad_attended)
        
        # Gradient through softmax (simplified)
        scale = 1.0 / math.sqrt(self.embed_dim)
        grad_scores = Matrix.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                for k in range(seq_len):
                    if j == k:
                        grad_scores[i][j] += grad_attention[i][k] * attention_weights[i][j] * (1 - attention_weights[i][j])
                    else:
                        grad_scores[i][j] -= grad_attention[i][k] * attention_weights[i][j] * attention_weights[i][k]
        grad_scores = Matrix.scalar_multiply(grad_scores, scale)
        
        # Gradient w.r.t. Q and K
        grad_Q = Matrix.multiply(grad_scores, K)
        grad_scores_T = Matrix.transpose(grad_scores)
        grad_K = Matrix.multiply(grad_scores_T, Q)
        
        # Gradient w.r.t. projections
        x_T = Matrix.transpose(x)
        self.grad_W_q = Matrix.multiply(x_T, grad_Q)
        self.grad_W_k = Matrix.multiply(x_T, grad_K)
        self.grad_W_v = Matrix.multiply(x_T, grad_V)
        
        # Gradient w.r.t. input
        W_q_T = Matrix.transpose(self.W_q)
        W_k_T = Matrix.transpose(self.W_k)
        W_v_T = Matrix.transpose(self.W_v)
        
        grad_x = Matrix.multiply(grad_Q, W_q_T)
        grad_x = Matrix.add(grad_x, Matrix.multiply(grad_K, W_k_T))
        grad_x = Matrix.add(grad_x, Matrix.multiply(grad_V, W_v_T))
        
        return grad_x
    
    def update(self, learning_rate):
        """Update weights using gradients."""
        for i in range(self.embed_dim):
            for j in range(self.embed_dim):
                self.W_q[i][j] -= learning_rate * self.grad_W_q[i][j]
                self.W_k[i][j] -= learning_rate * self.grad_W_k[i][j]
                self.W_v[i][j] -= learning_rate * self.grad_W_v[i][j]
                self.W_o[i][j] -= learning_rate * self.grad_W_o[i][j]
    
    def get_params(self):
        """Return parameters for saving."""
        return {
            "W_q": self.W_q,
            "W_k": self.W_k,
            "W_v": self.W_v,
            "W_o": self.W_o,
        }
    
    def set_params(self, params):
        """Load parameters."""
        self.W_q = params["W_q"]
        self.W_k = params["W_k"]
        self.W_v = params["W_v"]
        self.W_o = params["W_o"]


class FeedForward:
    """
    Feed-forward neural network layer with one hidden layer.
    """
    
    def __init__(self, embed_dim, hidden_dim=None, rng=None):
        """
        Initialize feed-forward layer.
        
        Args:
            embed_dim: Input/output dimension
            hidden_dim: Hidden layer dimension (default: 4 * embed_dim)
            rng: Random number generator
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or (4 * embed_dim)
        self.rng = rng or Random()
        
        # Two linear layers with GELU activation in between
        self.W1 = Matrix.random_init(embed_dim, self.hidden_dim, scale=1.0, rng=self.rng)
        self.b1 = [0.0 for _ in range(self.hidden_dim)]
        
        self.W2 = Matrix.random_init(self.hidden_dim, embed_dim, scale=1.0, rng=self.rng)
        self.b2 = [0.0 for _ in range(embed_dim)]
        
        # Gradient storage
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        
        # Cache
        self.cache = {}
    
    def forward(self, x):
        """
        Forward pass through feed-forward layer.
        
        Args:
            x: Input (seq_len x embed_dim)
            
        Returns:
            Output (seq_len x embed_dim)
        """
        seq_len = len(x)
        
        # First linear + bias
        hidden = Matrix.multiply(x, self.W1)
        for i in range(seq_len):
            for j in range(self.hidden_dim):
                hidden[i][j] += self.b1[j]
        
        # ReLU activation (simpler than GELU for pure Python)
        activated = Activations.relu(hidden)
        
        # Second linear + bias
        output = Matrix.multiply(activated, self.W2)
        for i in range(seq_len):
            for j in range(self.embed_dim):
                output[i][j] += self.b2[j]
        
        # Cache for backward
        self.cache = {
            "x": x,
            "hidden": hidden,
            "activated": activated,
        }
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through feed-forward layer.
        
        Args:
            grad_output: Gradient from next layer (seq_len x embed_dim)
            
        Returns:
            Gradient w.r.t. input (seq_len x embed_dim)
        """
        x = self.cache["x"]
        hidden = self.cache["hidden"]
        activated = self.cache["activated"]
        
        seq_len = len(x)
        
        # Gradient w.r.t. b2
        self.grad_b2 = [0.0 for _ in range(self.embed_dim)]
        for i in range(seq_len):
            for j in range(self.embed_dim):
                self.grad_b2[j] += grad_output[i][j]
        
        # Gradient w.r.t. W2
        activated_T = Matrix.transpose(activated)
        self.grad_W2 = Matrix.multiply(activated_T, grad_output)
        
        # Gradient w.r.t. activated
        W2_T = Matrix.transpose(self.W2)
        grad_activated = Matrix.multiply(grad_output, W2_T)
        
        # Gradient through ReLU
        relu_grad = Activations.relu_derivative(hidden)
        grad_hidden = Matrix.elementwise_multiply(grad_activated, relu_grad)
        
        # Gradient w.r.t. b1
        self.grad_b1 = [0.0 for _ in range(self.hidden_dim)]
        for i in range(seq_len):
            for j in range(self.hidden_dim):
                self.grad_b1[j] += grad_hidden[i][j]
        
        # Gradient w.r.t. W1
        x_T = Matrix.transpose(x)
        self.grad_W1 = Matrix.multiply(x_T, grad_hidden)
        
        # Gradient w.r.t. input
        W1_T = Matrix.transpose(self.W1)
        grad_x = Matrix.multiply(grad_hidden, W1_T)
        
        return grad_x
    
    def update(self, learning_rate):
        """Update weights using gradients."""
        for i in range(self.embed_dim):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= learning_rate * self.grad_W1[i][j]
        
        for j in range(self.hidden_dim):
            self.b1[j] -= learning_rate * self.grad_b1[j]
        
        for i in range(self.hidden_dim):
            for j in range(self.embed_dim):
                self.W2[i][j] -= learning_rate * self.grad_W2[i][j]
        
        for j in range(self.embed_dim):
            self.b2[j] -= learning_rate * self.grad_b2[j]
    
    def get_params(self):
        """Return parameters for saving."""
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
    
    def set_params(self, params):
        """Load parameters."""
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]


class LayerNorm:
    """
    Layer normalization.
    """
    
    def __init__(self, dim, eps=1e-5):
        """
        Initialize layer normalization.
        
        Args:
            dim: Dimension to normalize
            eps: Small constant for numerical stability
        """
        self.dim = dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = [1.0 for _ in range(dim)]  # Scale
        self.beta = [0.0 for _ in range(dim)]   # Shift
        
        # Gradient storage
        self.grad_gamma = None
        self.grad_beta = None
        
        # Cache
        self.cache = {}
    
    def forward(self, x):
        """
        Apply layer normalization.
        
        Args:
            x: Input (seq_len x dim)
            
        Returns:
            Normalized output (seq_len x dim)
        """
        seq_len = len(x)
        output = []
        
        means = []
        stds = []
        normalized = []
        
        for i in range(seq_len):
            # Compute mean and variance for this position
            row = x[i]
            mean = sum(row) / self.dim
            variance = sum((v - mean) ** 2 for v in row) / self.dim
            std = math.sqrt(variance + self.eps)
            
            means.append(mean)
            stds.append(std)
            
            # Normalize
            norm_row = [(v - mean) / std for v in row]
            normalized.append(norm_row)
            
            # Scale and shift
            out_row = [norm_row[j] * self.gamma[j] + self.beta[j] for j in range(self.dim)]
            output.append(out_row)
        
        # Cache for backward
        self.cache = {
            "x": x,
            "means": means,
            "stds": stds,
            "normalized": normalized,
        }
        
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through layer normalization.
        
        Args:
            grad_output: Gradient from next layer (seq_len x dim)
            
        Returns:
            Gradient w.r.t. input (seq_len x dim)
        """
        x = self.cache["x"]
        means = self.cache["means"]
        stds = self.cache["stds"]
        normalized = self.cache["normalized"]
        
        seq_len = len(x)
        
        # Gradient w.r.t. gamma and beta
        self.grad_gamma = [0.0 for _ in range(self.dim)]
        self.grad_beta = [0.0 for _ in range(self.dim)]
        
        for i in range(seq_len):
            for j in range(self.dim):
                self.grad_gamma[j] += grad_output[i][j] * normalized[i][j]
                self.grad_beta[j] += grad_output[i][j]
        
        # Gradient w.r.t. input (simplified)
        grad_x = []
        for i in range(seq_len):
            grad_row = []
            for j in range(self.dim):
                # Approximate gradient
                grad_row.append(grad_output[i][j] * self.gamma[j] / stds[i])
            grad_x.append(grad_row)
        
        return grad_x
    
    def update(self, learning_rate):
        """Update parameters using gradients."""
        for j in range(self.dim):
            self.gamma[j] -= learning_rate * self.grad_gamma[j]
            self.beta[j] -= learning_rate * self.grad_beta[j]
    
    def get_params(self):
        """Return parameters for saving."""
        return {
            "gamma": self.gamma,
            "beta": self.beta,
        }
    
    def set_params(self, params):
        """Load parameters."""
        self.gamma = params["gamma"]
        self.beta = params["beta"]
