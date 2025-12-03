# src/networks/lstm.py - SIMPLIFIED VERSION
"""
LSTM-based policy network (Simplified to match original)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMPolicyNet(nn.Module):
    """LSTM-based policy network - Simplified version matching original"""
    
    def __init__(self,
                 vocab_size: int = 20,
                 embed_dim: int = 512,
                 observation_size: int = 10,
                 hidden_size: int = 512,
                 action_size: int = 6,
                 num_layers: int = 1,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.observation_size = observation_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=None,
        )
        
        # Learnable positional encodings for the K tokens inside an observation
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)
        
        # ConcatMLP-style aggregator (like original)
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim * observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Policy head (logits)
        self.head = nn.Linear(hidden_size, action_size)
        
        # Hidden state
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
    def reset_state(self):
        """Reset LSTM hidden state"""
        self.hidden_state = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor [batch, seq, K]
        
        Returns
        -------
        logits : Tensor [batch, seq, action_size]
        """
        B, T, K = x.shape
        
        # Ensure input is LongTensor
        if x.dtype != torch.long:
            x = x.long()
        
        # Embed tokens: [B, T, K, D]
        x_embed = self.embedding(x)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_embed  # broadcast (K, D) -> (B, T, K, D)
        
        # Flatten and aggregate: [B, T, K*D] -> [B, T, H]
        x_flat = x_embed.view(B, T, -1)
        aggregated = self.aggregator(x_flat)
        
        # LSTM over the temporal dimension
        if self.hidden_state is None:
            # Initialize hidden state
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros_like(h0)
            self.hidden_state = (h0, c0)
        
        out, self.hidden_state = self.lstm(aggregated, self.hidden_state)
        
        # Get logits
        logits = self.head(out)
        
        return logits