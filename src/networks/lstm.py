"""
LSTM-based policy network
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .base import BaseNetwork, EmbeddingLayer, MLPAggregator


class LSTMPolicyNet(BaseNetwork):
    """LSTM-based policy network"""
    
    def __init__(self,
                 vocab_size: int = 20,
                 embed_dim: int = 512,
                 observation_size: int = 10,
                 hidden_size: int = 512,
                 action_size: int = 6,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 use_auxiliary: bool = False):
        
        super().__init__(observation_size, action_size, hidden_size, use_auxiliary)
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
        # Aggregator: K tokens -> 1 vector
        self.aggregator = MLPAggregator(
            observation_size * embed_dim, 
            hidden_size
        )
        
        # LSTM memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Auxiliary heads (optional)
        if use_auxiliary:
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1)
            )
            
            self.observation_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, observation_size * vocab_size)
            )
        
        # State
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
    def forward(self, 
                x: torch.Tensor,
                return_auxiliary: bool = False) -> torch.Tensor:
        """Forward pass"""
        B, T, K = x.shape
        
        # Ensure input is LongTensor for embedding layer
        if x.dtype != torch.long:
            x = x.long()
        
        # Embed and aggregate
        embedded = self.embedding(x)  # [B, T, K, D]
        aggregated = embedded.view(B, T, -1)  # [B, T, K*D]
        aggregated = self.aggregator(aggregated)  # [B, T, H]
        
        # LSTM over the temporal dimension
        if self.hidden_state is None:
            # Initialize hidden state with current batch size
            self.hidden_state = self._init_hidden(B, x.device)
        elif self.hidden_state[0].size(1) != B:
            # If batch size changed, reinitialize hidden state
            self.hidden_state = self._init_hidden(B, x.device)
        
        out, self.hidden_state = self.lstm(aggregated, self.hidden_state)  # out: [B, T, H]
        return self.head(out)
    
    def reset_state(self, batch_size: int = 1):
        """Reset LSTM hidden state"""
        self.hidden_state = None
        self._current_batch_size = batch_size  # Store the batch size

    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        # Use stored batch size or current batch size
        batch_size = getattr(self, '_current_batch_size', batch_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0
    
    def _get_config(self):
        """Get configuration for saving"""
        config = super()._get_config()
        config.update({
            'vocab_size': self.embedding.embedding.num_embeddings,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
        })
        return config