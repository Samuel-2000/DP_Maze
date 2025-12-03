"""
Multi-memory network combining LSTM, Transformer, and cache
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .base import BaseNetwork, EmbeddingLayer, AttentionAggregator
from .lstm import LSTMPolicyNet
from .transformer import TransformerPolicyNet


class NeuralCache(nn.Module):
    """Neural cache with content-based addressing"""
    
    def __init__(self, embed_dim: int, cache_size: int = 50):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cache_size = cache_size
        
        # Cache storage
        self.register_buffer('keys', torch.zeros(cache_size, embed_dim))
        self.register_buffer('values', torch.zeros(cache_size, embed_dim))
        self.register_buffer('usage', torch.zeros(cache_size))
        
        # Projections for addressing
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Usage decay
        self.decay = 0.95
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from cache"""
        B = query.shape[0]
        
        # Project query
        query_proj = self.query_proj(query)  # [B, D]
        keys_proj = self.key_proj(self.keys)  # [C, D]
        
        # Compute similarity
        scores = torch.matmul(query_proj, keys_proj.T) / (self.embed_dim ** 0.5)
        
        # Apply softmax with temperature
        attn = torch.softmax(scores, dim=-1)  # [B, C]
        
        # Retrieve values
        retrieved = torch.matmul(attn, self.values)  # [B, D]
        
        # Update usage
        self.usage = self.usage * self.decay + attn.mean(dim=0).detach()
        
        return retrieved, attn
    
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Write to cache"""
        with torch.no_grad():
            # Find least used slot
            slot = torch.argmin(self.usage)
            
            # Write to cache
            self.keys[slot] = key.detach()
            self.values[slot] = value.detach()
            self.usage[slot] = 1.0  # Reset usage for new entry
    
    def reset(self):
        """Reset cache"""
        self.keys.zero_()
        self.values.zero_()
        self.usage.zero_()


class MultiMemoryPolicyNet(BaseNetwork):
    """Network with multiple memory systems"""
    
    def __init__(self,
                 vocab_size: int = 20,
                 embed_dim: int = 512,
                 observation_size: int = 10,
                 hidden_size: int = 512,
                 action_size: int = 6,
                 transformer_heads: int = 8,
                 transformer_layers: int = 3,
                 cache_size: int = 50,
                 use_auxiliary: bool = False):
        
        super().__init__(observation_size, action_size, hidden_size, use_auxiliary)
        
        self.embed_dim = embed_dim
        self.cache_size = cache_size
        
        # Shared embedding
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
        # Multiple memory systems
        self.lstm_memory = LSTMPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_size=action_size,
            use_auxiliary=False
        )
        
        self.transformer_memory = TransformerPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_size=action_size,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            use_auxiliary=False
        )
        
        self.neural_cache = NeuralCache(embed_dim, cache_size)
        
        # Memory fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Auxiliary heads
        if use_auxiliary:
            self.energy_head = nn.Linear(hidden_size, 1)
            self.observation_head = nn.Linear(hidden_size, observation_size)
        
        # Write buffer for cache
        self.write_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
    def forward(self,
                x: torch.Tensor,
                return_auxiliary: bool = False) -> torch.Tensor:
        """Forward pass through all memory systems"""
        B, T, K = x.shape

        # Ensure input is LongTensor for embedding layer
        if x.dtype != torch.long:
            x = x.long()
        
        # Get embeddings
        embedded = self.embedding(x)  # [B, T, K, D]
        
        # Aggregate tokens
        aggregated = embedded.mean(dim=2)  # [B, T, D]
        
        # LSTM memory
        lstm_out = self._forward_lstm(aggregated)
        
        # Transformer memory
        transformer_out = self._forward_transformer(aggregated)
        
        # Neural cache (use current observation)
        current_obs = aggregated[:, -1] if T > 1 else aggregated.squeeze(1)
        cache_out, cache_attn = self.neural_cache(current_obs)
        cache_out = cache_out.unsqueeze(1).expand(-1, T, -1)
        
        # Cache writing decision
        if self.training and T > 1:
            self._maybe_write_to_cache(current_obs, aggregated[:, -1], cache_attn)
        
        # Fuse memories
        combined = torch.cat([lstm_out, transformer_out, cache_out], dim=-1)
        fused = self.fusion(combined)
        
        # Policy
        logits = self.policy_head(fused)
        
        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(fused)
            obs_pred = self.observation_head(fused)
            return logits, energy_pred, obs_pred
        
        return logits
    
    def _forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM"""
        # Convert to token format expected by LSTM
        B, T, D = x.shape
        tokenized = torch.arange(10).unsqueeze(0).unsqueeze(0).expand(B, T, -1).to(x.device)
        
        lstm_out = self.lstm_memory(tokenized)
        return lstm_out
    
    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer"""
        # Convert to token format
        B, T, D = x.shape
        tokenized = torch.arange(10).unsqueeze(0).unsqueeze(0).expand(B, T, -1).to(x.device)
        
        transformer_out = self.transformer_memory(tokenized)
        return transformer_out
    
    def _maybe_write_to_cache(self,
                             key: torch.Tensor,
                             value: torch.Tensor,
                             cache_attn: torch.Tensor):
        """Decide whether to write to cache based on attention"""
        # Write if current observation isn't well represented in cache
        max_attn = cache_attn.max(dim=-1)[0].mean()
        if max_attn < 0.3:
            self.write_buffer.append((key.detach(), value.detach()))
    
    def flush_cache_buffer(self):
        """Write buffered items to cache"""
        for key, value in self.write_buffer:
            self.neural_cache.write(key, value)
        self.write_buffer.clear()
    
    def reset_state(self, batch_size: int = 1):
        """Reset all memory systems"""
        self.lstm_memory.reset_state(batch_size)
        self.transformer_memory.reset_state(batch_size)
        self.neural_cache.reset()
        self.write_buffer.clear()
    
    def _get_config(self):
        """Get configuration for saving"""
        config = super()._get_config()
        config.update({
            'vocab_size': self.embedding.embedding.num_embeddings,
            'embed_dim': self.embed_dim,
            'cache_size': self.cache_size,
            'transformer_heads': self.transformer_memory.num_heads,
            'transformer_layers': self.transformer_memory.num_layers,
        })
        return config