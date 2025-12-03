# memory_maze/advanced_memory.py
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import numpy as np

class TransformerMemory(nn.Module):
    """
    Transformer-based memory for long-term dependencies
    """
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mem_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mem_size = mem_size
        
        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(torch.randn(mem_size, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
    def reset_state(self):
        self.memory_state = None
        
    def forward(self, observations: torch.Tensor, memory: Optional[torch.Tensor] = None):
        # observations: [batch, seq_len, embed_dim]
        batch_size, seq_len, _ = observations.shape
        
        if memory is None:
            memory = self.memory_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Combine memory and observations
        combined = torch.cat([memory, observations], dim=1)
        combined = self.pos_encoding(combined)
        
        # Apply transformer
        output = self.transformer(combined)
        
        # Split back into memory and observation outputs
        new_memory = output[:, :self.mem_size]
        obs_output = output[:, self.mem_size:]
        
        return obs_output, new_memory

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class NeuralCache(nn.Module):
    """
    Neural cache memory with content-based addressing
    """
    def __init__(self, embed_dim: int, cache_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.cache_size = cache_size
        
        # Cache: keys and values
        self.register_buffer('cache_keys', torch.zeros(cache_size, embed_dim))
        self.register_buffer('cache_values', torch.zeros(cache_size, embed_dim))
        self.cache_usage = torch.zeros(cache_size)
        
        # Attention mechanism
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
    def reset_state(self):
        self.cache_keys.zero_()
        self.cache_values.zero_()
        self.cache_usage.zero_()
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: [batch, embed_dim]
        batch_size = query.shape[0]
        
        # Project query
        query_proj = self.query_proj(query)  # [batch, embed_dim]
        keys_proj = self.key_proj(self.cache_keys)  # [cache_size, embed_dim]
        
        # Compute attention scores
        scores = torch.matmul(query_proj, keys_proj.T) / math.sqrt(self.embed_dim)
        attention_weights = torch.softmax(scores, dim=-1)  # [batch, cache_size]
        
        # Retrieve from cache
        retrieved = torch.matmul(attention_weights, self.cache_values)  # [batch, embed_dim]
        
        return retrieved, attention_weights
    
    def update(self, new_key: torch.Tensor, new_value: torch.Tensor, importance: float = 1.0):
        """Update cache with new key-value pair"""
        with torch.no_grad():
            # Find least used slot
            if torch.all(self.cache_usage == 0):
                # Cache is empty, use first slot
                slot = 0
            else:
                slot = torch.argmin(self.cache_usage)
            
            # Update cache
            self.cache_keys[slot] = new_key.squeeze(0).detach()
            self.cache_values[slot] = new_value.squeeze(0).detach()
            self.cache_usage[slot] = importance