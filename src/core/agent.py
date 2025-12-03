# src/core/agent.py - SIMPLIFIED VERSION
"""
Agent class (Simplified)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from src.networks.lstm import LSTMPolicyNet


class Agent:
    """Agent that interacts with environment using a policy network"""
    
    def __init__(self,
                 observation_size: int = 10,
                 action_size: int = 6,
                 hidden_size: int = 512,
                 device: str = 'auto'):
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Create network (matching original)
        self.network = LSTMPolicyNet(
            vocab_size=20,  # Matches original
            embed_dim=768,  # Matches original
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_size=action_size
        )
        
        self.network.to(self.device)
        
    def act(self, 
            observation: np.ndarray,
            training: bool = False) -> int:
        """Select action based on observation"""
        with torch.set_grad_enabled(training):
            # Convert observation to LongTensor for embedding layer
            obs_tensor = torch.from_numpy(observation).long().unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
            obs_tensor = obs_tensor.to(self.device)
            
            # Get action logits
            logits = self.network(obs_tensor)  # [1, 1, action_size]
            
            # Remove sequence dimension
            logits = logits.squeeze(1)  # [1, action_size]
            
            if training:
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                # Take greedy action
                action = logits.argmax(dim=-1).item()
            
            return action
    
    def reset(self):
        """Reset agent state"""
        self.network.reset_state()
    
    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            'state_dict': self.network.state_dict(),
            'config': {
                'observation_size': self.network.observation_size,
                'vocab_size': self.network.vocab_size,
                'hidden_size': self.network.lstm.hidden_size,
                'action_size': self.head.out_features
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto'):
        """Load agent from file"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Create agent
        agent = cls(
            observation_size=config.get('observation_size', 10),
            action_size=config.get('action_size', 6),
            hidden_size=config.get('hidden_size', 512),
            device=device
        )
        
        # Load weights
        agent.network.load_state_dict(checkpoint['state_dict'])
        
        return agent
    
    def test(self, 
             env,
             episodes: int = 10,
             visualize: bool = False,
             save_video: bool = False) -> Dict[str, Any]:
        """Test agent performance"""
        self.network.eval()
        
        rewards = []
        success_flags = []
        steps_list = []
        
        if save_video:
            import cv2
            video_writer = None
        
        for episode in range(episodes):
            obs = env.reset()
            self.reset()
            
            episode_reward = 0
            steps = 0
            done = False
            
            frames = []
            
            while not done and steps < env.max_steps:
                # Get action
                action = self.act(obs, training=False)
                
                # Take step
                obs, reward, done, _ = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # Record frame if needed
                if visualize or save_video:
                    frame = env.render(mode='rgb_array')
                    if save_video:
                        frames.append(frame)
                    if visualize:
                        cv2.imshow('Test', frame)
                        cv2.waitKey(1)
            
            # Save video if requested
            if save_video and frames:
                if video_writer is None:
                    h, w, _ = frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_path = f'results/videos/test_episode_{episode}.mp4'
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, 20.0, (w, h)
                    )
                
                for frame in frames:
                    video_writer.write(frame)
            
            rewards.append(episode_reward)
            success_flags.append(steps == env.max_steps)
            steps_list.append(steps)
        
        if save_video and video_writer is not None:
            video_writer.release()
        
        if visualize:
            cv2.destroyAllWindows()
        
        return {
            'rewards': rewards,
            'success_flags': success_flags,
            'steps': steps_list,
            'avg_reward': np.mean(rewards),
            'success_rate': np.mean(success_flags) * 100,
            'avg_steps': np.mean(steps_list),
            'std_reward': np.std(rewards)
        }