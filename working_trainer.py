# working_trainer.py
"""
Working trainer with proper tensor handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Creating working trainer...")

# First, let's recreate the exact network from the original working code
class SimpleLSTMPolicyNet(nn.Module):
    """Simple LSTM policy network matching original"""
    
    def __init__(self, vocab_size=20, embed_dim=768, observation_size=10, 
                 hidden_size=768, action_size=6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.observation_size = observation_size
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)
        
        # Aggregator (ConcatMLP)
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim * observation_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Policy head
        self.head = nn.Linear(hidden_size, action_size)
        
        # Hidden state
        self.hidden_state = None
    
    def reset_state(self):
        self.hidden_state = None
    
    def forward(self, x):
        B, T, K = x.shape
        
        # Embed
        embedded = self.embedding(x)  # [B, T, K, D]
        embedded = embedded + self.pos_embed  # Add positional encoding
        
        # Aggregate
        aggregated = embedded.view(B, T, -1)  # [B, T, K*D]
        aggregated = self.aggregator(aggregated)  # [B, T, D]
        
        # LSTM
        if self.hidden_state is None:
            h0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(1, B, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)
        
        out, self.hidden_state = self.lstm(aggregated, self.hidden_state)
        
        # Policy
        logits = self.head(out)
        
        return logits


# Simple environment simulator
class SimpleEnv:
    """Simple environment that mimics the original"""
    
    def __init__(self):
        self.observation_space = type('obj', (object,), {'shape': (10,)})
        self.action_space = type('obj', (object,), {'n': 6})
        self.max_steps = 100
    
    def reset(self):
        # Return observation like the original: 10 integers 0-19
        obs = np.zeros(10, dtype=np.int32)
        # First 8 are neighbor tiles (0-4)
        obs[:8] = np.random.randint(0, 5, size=8)
        # Last action (6-11)
        obs[8] = np.random.randint(6, 12)
        # Energy level (12-19)
        obs[9] = np.random.randint(12, 20)
        return obs, {}
    
    def step(self, action):
        # Simple simulation
        obs = self.reset()[0]
        # Reward: small positive for survival, occasional food
        reward = 0.01
        if np.random.random() > 0.9:  # 10% chance of finding food
            reward += 1.0
        
        # Done: 5% chance per step
        done = np.random.random() > 0.95
        
        return obs, reward, done, False, {}


def pad_batch(batch, max_len, pad_value=0, obs_dim=10):
    """Pad a batch of sequences to same length"""
    batch_size = len(batch)
    
    # Create padded array
    padded = np.full((batch_size, max_len, obs_dim), pad_value, dtype=np.int32)
    
    for i, seq in enumerate(batch):
        seq_len = min(len(seq), max_len)
        for t in range(seq_len):
            padded[i, t] = seq[t]
    
    return torch.tensor(padded, dtype=torch.long)


def compute_advantages(rewards, gamma=0.97):
    """Compute discounted returns and advantages"""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    for i in range(B):
        # Compute returns
        running_return = 0
        for t in reversed(range(T)):
            if rewards[i, t] != 0:  # Only for actual steps
                running_return = rewards[i, t] + gamma * running_return
                returns[i, t] = running_return
        
        # Simple baseline: mean return
        episode_returns = returns[i][rewards[i] != 0]
        if len(episode_returns) > 0:
            baseline = episode_returns.mean()
            advantages[i][rewards[i] != 0] = episode_returns - baseline
    
    # Normalize advantages
    valid_advantages = advantages[advantages != 0]
    if len(valid_advantages) > 0:
        mean_adv = valid_advantages.mean()
        std_adv = valid_advantages.std()
        advantages[advantages != 0] = (valid_advantages - mean_adv) / (std_adv + 1e-8)
    
    return returns, advantages


def main():
    print("Starting working trainer...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    env = SimpleEnv()
    
    # Create network
    net = SimpleLSTMPolicyNet(
        vocab_size=20,
        embed_dim=128,  # Smaller for faster training
        observation_size=10,
        hidden_size=128,
        action_size=6
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    
    # Training parameters
    epochs = 100
    batch_size = 8
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}")
    
    for epoch in range(epochs):
        # Collect experiences
        observations = []
        actions = []
        rewards = []
        
        for _ in range(batch_size):
            net.reset_state()
            obs, _ = env.reset()
            
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            
            for step in range(env.max_steps):
                ep_obs.append(obs.copy())
                
                # Convert to tensor
                obs_tensor = torch.from_numpy(obs).long().unsqueeze(0).unsqueeze(0).to(device)
                
                # Get action
                with torch.no_grad():
                    logits = net(obs_tensor)
                    probs = F.softmax(logits.squeeze(1), dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                ep_actions.append(action)
                
                # Take step
                obs, reward, done, _, _ = env.step(action)
                ep_rewards.append(reward)
                
                if done:
                    break
            
            observations.append(ep_obs)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
        
        # Pad sequences
        max_len = max(len(r) for r in rewards)
        
        obs_tensor = pad_batch(observations, max_len).to(device)
        act_tensor = pad_batch(actions, max_len, pad_value=0).to(device)
        rew_tensor = pad_batch(rewards, max_len, pad_value=0.0).float().to(device)
        
        # Forward pass
        net.reset_state()
        logits = net(obs_tensor)
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        act_log_probs = log_probs.gather(-1, act_tensor.unsqueeze(-1)).squeeze(-1)
        
        # Compute advantages
        returns, advantages = compute_advantages(rew_tensor)
        
        # Mask invalid steps
        mask = (rew_tensor != 0).float()
        valid_count = mask.sum().item()
        
        if valid_count > 0:
            policy_loss = -(act_log_probs * advantages.detach() * mask).sum() / valid_count
            
            # Entropy
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            entropy_loss = -0.01 * (entropy * mask).sum() / valid_count
            
            loss = policy_loss + entropy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            avg_reward = rew_tensor.sum(1).mean().item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}, "
                      f"Avg reward = {avg_reward:.3f}, "
                      f"Steps = {valid_count/batch_size:.1f}")
        else:
            print(f"Epoch {epoch + 1}: No valid steps")
    
    print("\nTraining completed!")
    
    # Save model
    torch.save({
        'state_dict': net.state_dict(),
        'config': {
            'vocab_size': 20,
            'embed_dim': 128,
            'observation_size': 10,
            'hidden_size': 128,
            'action_size': 6
        }
    }, 'models/working_model.pt')
    
    print("Model saved to models/working_model.pt")


if __name__ == "__main__":
    main()