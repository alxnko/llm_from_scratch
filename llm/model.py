"""
Simple Language Model combining all layers.
"""

import json
from .vocabulary import Vocabulary
from .math_utils import Matrix, Activations, Random
from .layers import Embedding, PositionalEncoding, SelfAttention, FeedForward, LayerNorm


class SimpleLLM:
    """
    A simple transformer-like language model.
    
    Architecture:
    - Token Embedding
    - Positional Encoding
    - N x (Self-Attention + Feed-Forward with residual connections and layer norm)
    - Output projection to vocabulary
    """
    
    def __init__(self, vocab_size, embed_dim=64, num_layers=2, max_seq_len=32, seed=None):
        """
        Initialize the language model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Random number generator
        self.rng = Random(seed)
        
        # Initialize layers
        self.embedding = Embedding(vocab_size, embed_dim, rng=self.rng)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.attention_layers = []
        self.ff_layers = []
        self.ln1_layers = []
        self.ln2_layers = []
        
        for _ in range(num_layers):
            self.attention_layers.append(SelfAttention(embed_dim, rng=self.rng))
            self.ff_layers.append(FeedForward(embed_dim, embed_dim * 4, rng=self.rng))
            self.ln1_layers.append(LayerNorm(embed_dim))
            self.ln2_layers.append(LayerNorm(embed_dim))
        
        # Output projection to vocabulary size
        self.output_proj = Matrix.random_init(embed_dim, vocab_size, scale=1.0, rng=self.rng)
        self.output_bias = [0.0 for _ in range(vocab_size)]
        
        # Gradient storage for output layer
        self.grad_output_proj = None
        self.grad_output_bias = None
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, token_indices, return_all_positions=False):
        """
        Forward pass through the model.
        
        Args:
            token_indices: List of token indices
            return_all_positions: If True, return logits for all positions
            
        Returns:
            Logits (vocabulary probabilities) for next token prediction
        """
        # Embedding lookup
        x = self.embedding.forward(token_indices)
        
        # Add positional encoding
        x = self.pos_encoding.forward(x)
        
        # Store for residual connections
        residuals = []
        
        # Pass through transformer layers
        for i in range(self.num_layers):
            residuals.append(x)
            
            # Self-attention with residual
            attn_out = self.attention_layers[i].forward(x)
            x = Matrix.add(x, attn_out)  # Residual connection
            x = self.ln1_layers[i].forward(x)
            
            # Feed-forward with residual
            ff_in = x
            ff_out = self.ff_layers[i].forward(x)
            x = Matrix.add(ff_in, ff_out)  # Residual connection
            x = self.ln2_layers[i].forward(x)
        
        # Cache for backward
        self.cache = {
            "residuals": residuals,
            "final_hidden": x,
        }
        
        # Output projection
        if return_all_positions:
            # Return logits for all positions
            logits = Matrix.multiply(x, self.output_proj)
            for i in range(len(logits)):
                for j in range(self.vocab_size):
                    logits[i][j] += self.output_bias[j]
        else:
            # Only return logits for last position
            last_hidden = x[-1]
            logits = []
            for j in range(self.vocab_size):
                val = sum(last_hidden[k] * self.output_proj[k][j] for k in range(self.embed_dim))
                logits.append(val + self.output_bias[j])
        
        return logits
    
    def predict_next(self, token_indices, temperature=1.0):
        """
        Predict the next token given a sequence.
        
        Args:
            token_indices: List of token indices
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Tuple of (predicted token index, probabilities)
        """
        logits = self.forward(token_indices)
        
        # Apply temperature
        if temperature != 1.0:
            logits = [l / temperature for l in logits]
        
        # Convert to probabilities
        probs = Activations.softmax(logits)
        
        # Sample from distribution
        r = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return i, probs
        
        return len(probs) - 1, probs
    
    def backward(self, grad_logits):
        """
        Backward pass through the model.
        
        Args:
            grad_logits: Gradient w.r.t. output logits
        """
        final_hidden = self.cache["final_hidden"]
        seq_len = len(final_hidden)
        
        # Gradient w.r.t. output projection
        # grad_logits is 1D (for last position only) or 2D (for all positions)
        if not isinstance(grad_logits[0], list):
            # Single position - expand to match hidden state
            grad_logits = [grad_logits]
            use_last_only = True
        else:
            use_last_only = False
        
        # Gradient w.r.t. output projection and bias
        self.grad_output_proj = Matrix.zeros(self.embed_dim, self.vocab_size)
        self.grad_output_bias = [0.0 for _ in range(self.vocab_size)]
        
        if use_last_only:
            # Only compute gradient for last position
            last_hidden = final_hidden[-1]
            for k in range(self.embed_dim):
                for j in range(self.vocab_size):
                    self.grad_output_proj[k][j] += last_hidden[k] * grad_logits[0][j]
            for j in range(self.vocab_size):
                self.grad_output_bias[j] += grad_logits[0][j]
            
            # Gradient w.r.t. final hidden (only last position)
            grad_hidden = Matrix.zeros(seq_len, self.embed_dim)
            for k in range(self.embed_dim):
                for j in range(self.vocab_size):
                    grad_hidden[-1][k] += grad_logits[0][j] * self.output_proj[k][j]
        else:
            hidden_T = Matrix.transpose(final_hidden)
            self.grad_output_proj = Matrix.multiply(hidden_T, grad_logits)
            for i in range(len(grad_logits)):
                for j in range(self.vocab_size):
                    self.grad_output_bias[j] += grad_logits[i][j]
            
            output_proj_T = Matrix.transpose(self.output_proj)
            grad_hidden = Matrix.multiply(grad_logits, output_proj_T)
        
        # Backward through transformer layers (in reverse)
        for i in range(self.num_layers - 1, -1, -1):
            # Backward through second layer norm
            grad_hidden = self.ln2_layers[i].backward(grad_hidden)
            
            # Backward through feed-forward (with residual)
            grad_ff = self.ff_layers[i].backward(grad_hidden)
            grad_hidden = Matrix.add(grad_hidden, grad_ff)
            
            # Backward through first layer norm
            grad_hidden = self.ln1_layers[i].backward(grad_hidden)
            
            # Backward through self-attention (with residual)
            grad_attn = self.attention_layers[i].backward(grad_hidden)
            grad_hidden = Matrix.add(grad_hidden, grad_attn)
        
        # Backward through embedding
        self.embedding.backward(grad_hidden)
    
    def update(self, learning_rate):
        """Update all model parameters."""
        # Update embedding
        self.embedding.update(learning_rate)
        
        # Update transformer layers
        for i in range(self.num_layers):
            self.attention_layers[i].update(learning_rate)
            self.ff_layers[i].update(learning_rate)
            self.ln1_layers[i].update(learning_rate)
            self.ln2_layers[i].update(learning_rate)
        
        # Update output projection
        for k in range(self.embed_dim):
            for j in range(self.vocab_size):
                self.output_proj[k][j] -= learning_rate * self.grad_output_proj[k][j]
        for j in range(self.vocab_size):
            self.output_bias[j] -= learning_rate * self.grad_output_bias[j]
    
    def save(self, filepath):
        """Save model parameters to JSON file."""
        params = {
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "max_seq_len": self.max_seq_len,
            },
            "embedding": self.embedding.get_params(),
            "attention_layers": [layer.get_params() for layer in self.attention_layers],
            "ff_layers": [layer.get_params() for layer in self.ff_layers],
            "ln1_layers": [layer.get_params() for layer in self.ln1_layers],
            "ln2_layers": [layer.get_params() for layer in self.ln2_layers],
            "output_proj": self.output_proj,
            "output_bias": self.output_bias,
        }
        
        with open(filepath, "w") as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from JSON file."""
        with open(filepath, "r") as f:
            params = json.load(f)
        
        config = params["config"]
        model = cls(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            max_seq_len=config["max_seq_len"],
        )
        
        model.embedding.set_params(params["embedding"])
        
        for i, layer_params in enumerate(params["attention_layers"]):
            model.attention_layers[i].set_params(layer_params)
        
        for i, layer_params in enumerate(params["ff_layers"]):
            model.ff_layers[i].set_params(layer_params)
        
        for i, layer_params in enumerate(params["ln1_layers"]):
            model.ln1_layers[i].set_params(layer_params)
        
        for i, layer_params in enumerate(params["ln2_layers"]):
            model.ln2_layers[i].set_params(layer_params)
        
        model.output_proj = params["output_proj"]
        model.output_bias = params["output_bias"]
        
        return model
