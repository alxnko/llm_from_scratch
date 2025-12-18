"""
LLM From Scratch - Pure Python Implementation
A minimal language model with no external dependencies.
"""

from .vocabulary import Vocabulary, GrammarRule
from .math_utils import Matrix, Activations, Random
from .layers import Embedding, SelfAttention, FeedForward, LayerNorm
from .model import SimpleLLM
from .trainer import Trainer
from .generator import SentenceGenerator

__version__ = "1.0.0"
__all__ = [
    "Vocabulary",
    "GrammarRule", 
    "Matrix",
    "Activations",
    "Random",
    "Embedding",
    "SelfAttention",
    "FeedForward",
    "LayerNorm",
    "SimpleLLM",
    "Trainer",
    "SentenceGenerator",
]
