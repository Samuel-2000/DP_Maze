# memory_maze/simple_training.py (updated)
import argparse
import copy
import cv2
from time import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from memory_maze import GridMazeWorld
from torch import nn
import math


# Import advanced memory modules
try:
    from .advanced_memory import TransformerMemory, NeuralCache, PositionalEncoding
except ImportError:
    # For direct execution
    import sys
    sys.path.append('.')
    from advanced_memory import TransformerMemory, NeuralCache, PositionalEncoding

def parse_args():
    parser = argparse.ArgumentParser("Train a maze agent using simple policy gradient method.")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument("--max-age", type=int, default=50)
    parser.add_argument("--net-path", type=str)
    parser.add_argument("--env-group-size", type=int, default=1,
                        help="Number of the same environments - used to reduce variance in training. Advantages are computed per group.")
    parser.add_argument("--max-epoch", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--network-type", choices=['lstm', 'transformer', 'multimemory'], default='lstm',
                        help="Type of memory network to use")
    parser.add_argument("--auxiliary-tasks", action='store_true',
                        help="Use auxiliary tasks for training")
    parser.add_argument("--cache-size", type=int, default=50,
                        help="Size of neural cache (for multimemory network)")
    parser.add_argument("--transformer-heads", type=int, default=8,
                        help="Number of attention heads for transformer")
    parser.add_argument("--transformer-layers", type=int, default=3,
                        help="Number of transformer layers")
    return parser.parse_args()


class AttnAggregator(nn.Module):
    """
    Attention‑style pooling that turns a set of token embeddings
    E ∈ ℝ^{B×T×K×D}  (B=batch, T=seq, K=observation_size, D=emb dim)
    into a single vector per observation:  ℝ^{B×T×D}.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)  # learnable query

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, K, D]  →  scores: [batch, seq, K, 1]
        scores = self.score(torch.tanh(x))
        α = torch.softmax(scores, dim=2)                  # weights along K
        pooled = (α * x).sum(dim=2)                       # [batch, seq, D]
        return pooled


class ConcatMLPAggregator(nn.Module):
    """
    ConcatMLP-style pooling that turns a set of token embeddings
    E ∈ ℝ^{B×T×K×D}  (B=batch, T=seq, K=observation_size, D=emb dim)
    into a single vector per observation:  ℝ^{B×T×D}.

    D - embed_dim
    K - embed_count
    """
    def __init__(self, embed_dim: int, embed_count: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_count = embed_count
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * embed_count, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, K, D]  →  x: [batch, seq, K*D]
        x = x.view(x.shape[0], x.shape[1], -1)
        # x: [batch, seq, K*D]  →  x: [batch, seq, D]
        x = self.mlp(x)
        return x


class TransformerPolicyNet(nn.Module):
    """
    Policy network with Transformer memory instead of LSTM
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        observation_size: int,
        hidden_size: int,
        action_count: int,
        num_heads: int = 8,
        num_layers: int = 3,
        mem_size: int = 10,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.embed_dim = embed_dim
        
        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        
        # Learnable positional encodings
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)
        
        # Aggregator
        self.aggregator = ConcatMLPAggregator(embed_dim, observation_size)
        
        # Transformer memory
        self.transformer_memory = TransformerMemory(
            embed_dim, num_heads, num_layers, mem_size
        )
        
        # Policy head
        self.head = nn.Linear(embed_dim, action_count)
        
        # Auxiliary heads for self-supervised learning
        self.use_auxiliary = False
        self.energy_predictor = nn.Linear(embed_dim, 1)
        self.obs_predictor = nn.Linear(embed_dim, observation_size * vocab_size)
        
    def reset_state(self):
        self.transformer_memory.reset_state()
        
    def forward(self, obs: torch.Tensor, return_auxiliary: bool = False):
        B, T, K = obs.shape
        
        # Embed observations
        x = self.embedding(obs)  # [B, T, K, D]
        x = x + self.pos_embed
        x_agg = self.aggregator(x)  # [B, T, D]
        
        # Apply transformer memory
        transformer_out, _ = self.transformer_memory(x_agg)
        
        # Get last output for decision making
        if T > 1:
            current_state = transformer_out[:, -1:]
        else:
            current_state = transformer_out
            
        logits = self.head(current_state)
        
        if return_auxiliary and self.use_auxiliary:
            # Auxiliary predictions
            energy_pred = self.energy_predictor(transformer_out)
            obs_pred = self.obs_predictor(transformer_out)
            obs_pred = obs_pred.view(B, T, self.observation_size, -1)
            return logits, energy_pred, obs_pred
        
        return logits


class MultiMemoryPolicyNet(nn.Module):
    """
    Policy network with multiple memory systems:
    - LSTM for short-term memory
    - Transformer for long-term dependencies  
    - Neural cache for important events
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        observation_size: int,
        hidden_size: int,
        action_count: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        cache_size: int = 50,
        use_auxiliary: bool = False
    ):
        super().__init__()
        self.observation_size = observation_size
        self.use_auxiliary = use_auxiliary
        
        # Embedding and aggregation
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)
        
        self.aggregator = ConcatMLPAggregator(embed_dim, observation_size)
        
        # Multiple memory systems
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.transformer_memory = TransformerMemory(
            embed_dim, transformer_heads, transformer_layers, mem_size=10
        )
        self.neural_cache = NeuralCache(embed_dim, cache_size)
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(hidden_size + embed_dim + embed_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        # Policy head
        self.head = nn.Linear(hidden_size, action_count)
        
        # Auxiliary prediction heads
        if use_auxiliary:
            self.energy_predictor = nn.Linear(hidden_size, 1)
            self.obs_predictor = nn.Linear(hidden_size, observation_size)
        
        # State tracking
        self.hidden_state = None
        self.cache_updates = []
        
    def reset_state(self):
        self.hidden_state = None
        self.transformer_memory.reset_state()
        self.neural_cache.reset_state()
        self.cache_updates = []
        
    def forward(self, obs: torch.Tensor, return_auxiliary: bool = False):
        B, T, K = obs.shape
        
        # Embed observations
        x = self.embedding(obs)  # [B, T, K, D]
        x = x + self.pos_embed
        x_agg = self.aggregator(x)  # [B, T, D]
        
        # LSTM memory
        if self.hidden_state is None:
            h0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)
            
        lstm_out, self.hidden_state = self.lstm(x_agg, self.hidden_state)
        
        # Transformer memory
        transformer_out, _ = self.transformer_memory(x_agg)
        
        # Neural cache (use current observation as query)
        if T > 0:
            current_obs = x_agg[:, -1] if T > 1 else x_agg.squeeze(1)
            cache_retrieved, cache_attention = self.neural_cache(current_obs)
            cache_retrieved = cache_retrieved.unsqueeze(1)
            
            # Store important observations in cache
            if self.training and T > 1:
                # Use attention weights to determine importance
                importance = cache_attention.max(dim=1)[0].mean().item()
                if importance < 0.3:  # If current observation isn't well represented in cache
                    self.cache_updates.append((current_obs.detach(), x_agg[:, -1].detach()))
        
        # Fuse memories
        if T > 1:
            # For sequence length > 1, fuse all three
            fused = self.memory_fusion(
                torch.cat([lstm_out, transformer_out, cache_retrieved], dim=-1)
            )
        else:
            # For single step, use LSTM output and current embedding
            fused = self.memory_fusion(
                torch.cat([lstm_out, x_agg, cache_retrieved], dim=-1)
            )
        
        logits = self.head(fused)
        
        if return_auxiliary and self.use_auxiliary:
            # Auxiliary predictions
            energy_pred = self.energy_predictor(fused)
            obs_pred = self.obs_predictor(fused)
            return logits, energy_pred, obs_pred
        
        return logits
    
    def update_cache(self):
        """Update neural cache with stored observations"""
        if self.cache_updates and self.training:
            for key, value in self.cache_updates:
                self.neural_cache.update(key, value)
            self.cache_updates = []


class MemoryPolicyNet(nn.Module):
    """
    Original LSTM-based policy network (unchanged for backward compatibility)
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        observation_size: int,
        hidden_size: int,
        action_count: int,
        num_layers: int = 1,
        pad_idx: int = None,
    ):
        super().__init__()
        self.observation_size = observation_size

        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # Learnable positional encodings for the K tokens inside an observation
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)

        # Attention aggregator (K tokens → 1 vector)
        self.aggregator = ConcatMLPAggregator(embed_dim, observation_size)

        # Temporal memory across the seq dimension
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Policy head (logits)
        self.head = nn.Linear(hidden_size, action_count)
        self.hidden_state: tuple[torch.Tensor, torch.Tensor] = None

    def reset_state(self):
        self.hidden_state = None

    def forward(
        self,
        obs: torch.Tensor,                           # [batch, seq, K]  (Long)
    ) -> torch.Tensor:
        B, T, K = obs.shape
        assert K == self.observation_size, "Mismatch in observation_size"

        x = self.embedding(obs)                       # [B, T, K, D]
        x = x + self.pos_embed                        # broadcast (K, D) → (B,T,K,D)
        x = self.aggregator(x)                        # [B, T, D]

        # LSTM over the temporal dimension
        if self.hidden_state is None:
            # Initialize hidden state
            self.hidden_state = self.initial_state(B, device=x.device)

        out, self.hidden_state = self.lstm(x, self.hidden_state)       # out: [B, T, H]
        return self.head(out)

    def initial_state(self, batch_size: int, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0


def create_network(args, vocab_size, observation_size, action_count):
    """Create network based on network type"""
    embed_dim = 512  # Reduced from 768 for faster training
    hidden_size = 512
    
    if args.network_type == 'transformer':
        net = TransformerPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count,
            num_heads=args.transformer_heads,
            num_layers=args.transformer_layers,
            mem_size=10
        )
        net.use_auxiliary = args.auxiliary_tasks
        print(f"Created Transformer network with {args.transformer_layers} layers, {args.transformer_heads} heads")
        
    elif args.network_type == 'multimemory':
        net = MultiMemoryPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            cache_size=args.cache_size,
            use_auxiliary=args.auxiliary_tasks
        )
        print(f"Created MultiMemory network with cache size {args.cache_size}")
        
    else:  # lstm (default)
        net = MemoryPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count
        )
        print("Created LSTM network")
    
    return net


def train_with_auxiliary_losses(
    net, 
    episode_observations, 
    episode_actions, 
    episode_rewards,
    episode_energies,
    optimizer,
    args,
    device
):
    """Training with policy loss + entropy + auxiliary losses"""
    B, T, K = episode_observations.shape
    
    # Reset network state
    net.reset_state()
    
    # Forward pass
    if args.auxiliary_tasks and hasattr(net, 'use_auxiliary') and net.use_auxiliary:
        outputs = net(episode_observations, return_auxiliary=True)
        if args.network_type == 'transformer':
            logits, energy_pred, obs_pred = outputs
        else:  # multimemory
            logits, energy_pred, obs_pred = outputs
    else:
        logits = net(episode_observations)
    
    # Policy loss
    log_probs = F.log_softmax(logits, dim=-1)
    act_log_probs = log_probs.gather(-1, episode_actions.unsqueeze(-1)).squeeze(-1)
    
    # Compute advantages
    with torch.no_grad():
        returns = torch.zeros_like(episode_rewards)
        running = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            running = episode_rewards[:, t] + args.gamma * running
            returns[:, t] = running
        
        if args.env_group_size > 1:
            advantages = torch.zeros_like(returns)
            for i in range(0, B, args.env_group_size):
                group = returns[i:i + args.env_group_size]
                mean = group.mean(dim=(0, 1), keepdim=True)
                std = group.std(dim=(0, 1), keepdim=True)
                group = (group - mean) / (std + 1e-8)
                advantages[i:i + args.env_group_size] = group
        else:
            baseline = returns.mean(dim=1, keepdim=True)
            advantages = returns - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    policy_loss = -(act_log_probs * advantages.detach()).mean()
    
    # Entropy regularization
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
    entropy_loss = -args.entropy_coef * entropy
    
    # Total loss starts with policy and entropy
    total_loss = policy_loss + entropy_loss
    
    # Add auxiliary losses if using them
    if args.auxiliary_tasks and hasattr(net, 'use_auxiliary') and net.use_auxiliary:
        # Energy prediction loss
        energy_target = episode_energies.unsqueeze(-1)  # [B, T, 1]
        energy_loss = F.mse_loss(energy_pred, energy_target)
        
        # Observation prediction loss (predict next observation)
        if T > 1:
            obs_target = episode_observations[:, 1:]  # Next observations
            obs_pred_current = obs_pred[:, :-1]  # Predictions for current state
            
            # Reshape for cross entropy
            obs_pred_reshaped = obs_pred_current.reshape(-1, K)
            obs_target_reshaped = obs_target.reshape(-1)
            
            obs_loss = F.cross_entropy(obs_pred_reshaped, obs_target_reshaped)
        else:
            obs_loss = torch.tensor(0.0, device=device)
        
        # Add auxiliary losses with coefficients
        total_loss = total_loss + 0.1 * energy_loss + 0.05 * obs_loss
    
    # Optimization
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
    optimizer.step()
    
    # Update cache if using multi-memory network
    if args.network_type == 'multimemory':
        net.update_cache()
    
    # Return metrics
    metrics = {
        'total_loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'entropy_loss': entropy_loss.item(),
    }
    
    if args.auxiliary_tasks and hasattr(net, 'use_auxiliary') and net.use_auxiliary:
        metrics['energy_loss'] = energy_loss.item() if 'energy_loss' in locals() else 0.0
        metrics['obs_loss'] = obs_loss.item() if 'obs_loss' in locals() else 0.0
    
    return metrics


def main():
    import numpy as np
    args = parse_args()
    
    # Add training hyperparameters to args
    args.gamma = 0.97
    args.entropy_coef = 0.01
    args.max_grad_norm = 1.0
    args.aux_coef = 0.1
    
    obstacle_fraction = 0.3
    grid_size = 11
    obstacle_count = int((grid_size - 2) * (grid_size - 2) * obstacle_fraction)
    food_source_count = 4
    food_energy = 10
    initial_energy = 30

    environments = [GridMazeWorld(
        max_age=args.max_age,
        grid_size=grid_size,
        obstacle_count=obstacle_count,
        food_source_count=food_source_count,
        food_energy=food_energy,
        initial_energy=initial_energy,
        name=str(i),
    ) for i in range(args.batch_size)]

    observation_size = 10
    vocab_size = 20
    action_count = 6  # Updated to match Actions enum

    # Create network
    net = create_network(args, vocab_size, observation_size, action_count)

    if args.net_path:
        checkpoint = torch.load(args.net_path)
        net.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.net_path}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)

    t0 = time()
    
    # Create outer progress bar for epochs (position 0, stays at top)
    outer_pbar = tqdm(
        total=args.max_epoch, 
        desc="🧠 Training Progress", 
        position=0, 
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    
    # Create inner progress bar for current interval (position 1, updates every iteration)
    inner_pbar = tqdm(
        total=args.save_interval,
        desc=f"📊 Epoch {0:06d}",
        position=1,
        leave=False,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
    )
    
    # Statistics tracking
    reward_history = []
    loss_history = []
    policy_loss_history = []
    entropy_loss_history = []
    speed_history = []
    
    # Best model tracking
    best_reward = -float('inf')
    best_epoch = 0
    
    # Training loop
    for epoch in range(args.max_epoch):
        t1 = time()
        
        with torch.no_grad():
            # Reset environments
            observations = [env.reset()[0] for env in environments]

            final_environments = []
            final_observations = []
            for env, obs in zip(environments, observations):
                for i in range(args.env_group_size):
                    final_environments.append(copy.deepcopy(env))
                    final_observations.append(obs)
            
            final_observations = torch.tensor(final_observations, dtype=torch.long).to(device).unsqueeze(1)
            net.reset_state()

            # Collect episodes with energies for auxiliary tasks
            episode_actions, episode_observations, episode_rewards, episode_energies = get_episodes(
                args, device, final_environments, final_observations, net
            )

            episode_observations = torch.cat(episode_observations, dim=1).to(device)
            episode_actions = torch.stack(episode_actions, dim=1).to(device)
            episode_rewards = torch.tensor(episode_rewards, dtype=torch.float).to(device).permute(1, 0)
            
            # Collect energies for auxiliary tasks
            if args.auxiliary_tasks:
                episode_energies = torch.tensor(episode_energies, dtype=torch.float).to(device).permute(1, 0)

        # Train with appropriate loss function
        if args.auxiliary_tasks:
            metrics = train_with_auxiliary_losses(
                net, episode_observations, episode_actions, 
                episode_rewards, episode_energies, optimizer, args, device
            )
        else:
            # Original training
            net.reset_state()
            logits = net(episode_observations)
            
            log_probs = F.log_softmax(logits, dim=-1)
            act_log_probs = log_probs.gather(-1, episode_actions.unsqueeze(-1)).squeeze(-1)
            
            with torch.no_grad():
                B, T = episode_rewards.shape
                returns = torch.zeros_like(episode_rewards)
                running = torch.zeros(B, device=device)
                for t in reversed(range(T)):
                    running = episode_rewards[:, t] + args.gamma * running
                    returns[:, t] = running
                
                if args.env_group_size > 1:
                    advantages = torch.zeros_like(returns)
                    for i in range(0, B, args.env_group_size):
                        group = returns[i:i + args.env_group_size]
                        mean = group.mean(dim=(0, 1), keepdim=True)
                        std = group.std(dim=(0, 1), keepdim=True)
                        group = (group - mean) / (std + 1e-8)
                        advantages[i:i + args.env_group_size] = group
                else:
                    baseline = returns.mean(dim=1, keepdim=True)
                    advantages = returns - baseline
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = -(act_log_probs * advantages.detach()).mean()
            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
            entropy_loss = -args.entropy_coef * entropy
            loss = policy_loss + entropy_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            
            metrics = {
                'total_loss': loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy_loss': entropy_loss.item()
            }
        
        # Compute statistics
        average_rewards = episode_rewards.sum(1).mean().item()
        alive_lengths = (episode_rewards > 0).sum(1).float().mean().item()
        epoch_time = time() - t1
        epoch_speed = 1.0 / epoch_time if epoch_time > 0 else 0
        
        # Store history
        reward_history.append(average_rewards)
        loss_history.append(metrics['total_loss'])
        policy_loss_history.append(metrics['policy_loss'])
        entropy_loss_history.append(metrics['entropy_loss'])
        speed_history.append(epoch_speed)
        
        # Keep only recent history (last 100 epochs)
        window_size = 100
        if len(reward_history) > window_size:
            reward_history = reward_history[-window_size:]
            loss_history = loss_history[-window_size:]
            policy_loss_history = policy_loss_history[-window_size:]
            entropy_loss_history = entropy_loss_history[-window_size:]
            speed_history = speed_history[-window_size:]
        
        # Update inner progress bar (every iteration)
        inner_pbar.update(1)
        
        # Prepare inner bar postfix with current stats
        inner_postfix = {
            'R': f"{average_rewards:6.2f}",
            'L': f"{alive_lengths:5.1f}",
            'loss': f"{metrics['total_loss']:7.4f}",
            'P': f"{metrics['policy_loss']:7.4f}",
            'E': f"{metrics['entropy_loss']:7.4f}",
        }
        
        # Add auxiliary losses if using them
        if args.auxiliary_tasks and hasattr(net, 'use_auxiliary') and net.use_auxiliary:
            if 'energy_loss' in metrics:
                inner_postfix['Ener'] = f"{metrics['energy_loss']:7.4f}"
            if 'obs_loss' in metrics:
                inner_postfix['Obs'] = f"{metrics['obs_loss']:7.4f}"
        
        inner_pbar.set_postfix(inner_postfix)
        
        # Update outer progress bar description and postfix (every 10 epochs)
        if epoch % 10 == 0:
            # Calculate rolling averages
            window = min(10, len(reward_history))
            avg_reward_window = np.mean(reward_history[-window:]) if reward_history else 0
            avg_loss_window = np.mean(loss_history[-window:]) if loss_history else 0
            avg_speed = np.mean(speed_history[-window:]) if speed_history else 0
            
            outer_postfix = {
                'avg_R': f"{avg_reward_window:6.2f}",
                'avg_loss': f"{avg_loss_window:7.4f}",
                'speed': f"{avg_speed:5.1f} ep/s",
                'best_R': f"{best_reward:6.2f}@{best_epoch}"
            }
            
            outer_pbar.set_postfix(outer_postfix)
        
        # Update outer bar
        outer_pbar.update(1)
        
        # Check for best model
        if average_rewards > best_reward:
            best_reward = average_rewards
            best_epoch = epoch
            
            # Save best model
            best_model_path = f"models/policy_{args.network_type}_best.pt"
            torch.save(net.state_dict(), best_model_path)
            
            # Update inner bar with star indicator
            inner_pbar.set_description(f"📊 Epoch {epoch:06d} ★")
        
        # Save model at intervals
        if (epoch + 1) % args.save_interval == 0 or epoch == args.max_epoch - 1:
            model_path = f"models/policy_{args.network_type}_epoch_{epoch:06d}.pt"
            torch.save(net.state_dict(), model_path)
            
            # Calculate interval statistics
            interval_start = max(0, epoch - args.save_interval + 1)
            interval_end = epoch + 1
            
            # Get metrics for this interval
            interval_rewards = reward_history[-args.save_interval:] if len(reward_history) >= args.save_interval else reward_history
            interval_losses = loss_history[-args.save_interval:] if len(loss_history) >= args.save_interval else loss_history
            
            avg_interval_reward = np.mean(interval_rewards) if interval_rewards else 0
            std_interval_reward = np.std(interval_rewards) if interval_rewards else 0
            avg_interval_loss = np.mean(interval_losses) if interval_losses else 0
            
            # Write interval summary to console (above progress bars)
            outer_pbar.write("─" * 80)
            outer_pbar.write(f"💾 SAVED: {model_path}")
            outer_pbar.write(f"   Interval [{interval_start:06d}-{interval_end:06d}] stats:")
            outer_pbar.write(f"   • Reward: {avg_interval_reward:6.2f} ± {std_interval_reward:5.2f}")
            outer_pbar.write(f"   • Loss:   {avg_interval_loss:7.4f}")
            outer_pbar.write(f"   • Current: R={average_rewards:6.2f}, L={alive_lengths:5.1f}, loss={metrics['total_loss']:7.4f}")
            
            # Reset inner progress bar for next interval
            inner_pbar.close()
            inner_pbar = tqdm(
                total=args.save_interval,
                desc=f"📊 Epoch {epoch+1:06d}",
                position=1,
                leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
            )
        
        # Early stopping check (optional)
        if len(reward_history) >= 100:
            # Check if no improvement in last 100 epochs
            recent_rewards = reward_history[-100:]
            if max(recent_rewards) == recent_rewards[0]:
                outer_pbar.write("⚠️  Early stopping: No improvement in last 100 epochs")
                break
    
    # Close progress bars
    inner_pbar.close()
    outer_pbar.close()
    
    # Print final training summary
    total_time = time() - t0
    avg_speed = args.max_epoch / total_time if total_time > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎉 TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total epochs:          {args.max_epoch:,}")
    print(f"Total time:            {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Average speed:         {avg_speed:.1f} ep/s")
    print(f"Final reward:          {reward_history[-1] if reward_history else 0:.2f}")
    print(f"Final loss:            {loss_history[-1] if loss_history else 0:.4f}")
    print(f"Best reward:           {best_reward:.2f} (epoch {best_epoch})")
    print(f"Average reward:        {np.mean(reward_history) if reward_history else 0:.2f}")
    print(f"Average loss:          {np.mean(loss_history) if loss_history else 0:.4f}")
    print(f"\n📁 Model files saved to: models/")
    print(f"   • Best model:        policy_{args.network_type}_best.pt")
    print(f"   • Final model:       policy_{args.network_type}_final.pt")
    print(f"   • Checkpoints:       policy_{args.network_type}_epoch_*.pt")
    print(f"{'='*80}")
    
    # Save final model
    final_model_path = f"models/policy_{args.network_type}_final.pt"
    torch.save(net.state_dict(), final_model_path)
    
    # Save training statistics
    if reward_history:
        stats = {
            'epochs': list(range(len(reward_history))),
            'rewards': reward_history,
            'losses': loss_history,
            'policy_losses': policy_loss_history,
            'entropy_losses': entropy_loss_history,
            'speeds': speed_history,
            'best_epoch': best_epoch,
            'best_reward': best_reward,
            'total_time': total_time,
            'network_type': args.network_type,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_age': args.max_age
        }
        
        # Save as numpy file
        import numpy as np
        stats_path = f"experiments/stats_{args.network_type}_{int(time())}.npz"
        os.makedirs('experiments', exist_ok=True)
        np.savez(stats_path, **stats)
        print(f"📊 Training statistics saved to: {stats_path}")


def get_episodes(args, device, environments, observations, net, show=False):
    """Collect episodes, optionally returning energies for auxiliary tasks"""
    import numpy as np
    episode_rewards = []
    episode_observations = []
    episode_actions = []
    episode_energies = []  # For auxiliary tasks
    
    
    # Create progress bar for episode steps if not in show mode
    if not show and args.max_age > 50:
        step_pbar = tqdm(
            total=args.max_age,
            desc="🎮 Episode Steps",
            position=2,
            leave=False,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]'
        )
    else:
        step_pbar = None
    
    for i in range(args.max_age):
        episode_observations.append(observations)
        logits = net(observations).squeeze(1)
        
        if show:
            sampled_actions = logits.argmax(dim=-1)
        else:
            sampled_actions = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            sampled_actions = sampled_actions.squeeze(1)
        
        episode_actions.append(sampled_actions)

        # Take action
        results = [env.step(action.item()) for env, action in zip(environments, sampled_actions)]
        
        if show:
            env = environments[0]
            last_image = env.render(mode="image")
            cv2.imshow(env.name, last_image)
            key = cv2.waitKey(20)
            if key == ord(' '):
                break

        observations_list, rewards, dones, infos = zip(*results)
        
        # Store energies if needed for auxiliary tasks
        if args.auxiliary_tasks:
            energies = [env.energy for env in environments]
            episode_energies.append(energies)
        
        episode_rewards.append(rewards)
        observations = torch.tensor(observations_list, dtype=torch.long)
        observations = observations.unsqueeze(1).to(device)

        # Update step progress bar
        if step_pbar:
            avg_reward = np.mean(rewards)
            step_pbar.update(1)
            step_pbar.set_postfix({'avg_R': f"{avg_reward:.2f}"})
        
        if all(dones):
            if step_pbar:
                step_pbar.n = step_pbar.total  # Complete the bar
                step_pbar.refresh()
            break
    
    if show:
        for i in range(30):
            image = last_image if i % 2 == 0 else last_image * 0
            cv2.imshow(env.name, image)
            cv2.waitKey(20)
    
    # Close step progress bar
    if step_pbar:
        step_pbar.close()
    
    return episode_actions, episode_observations, episode_rewards, episode_energies


if __name__ == "__main__":
    main()