"""
Agent class for interacting with environment
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from src.networks import (
    LSTMPolicyNet, 
    TransformerPolicyNet, 
    MultiMemoryPolicyNet
)


class Agent:
    """Agent that interacts with environment using a policy network"""
    
    def __init__(self,
                 network_type: str = 'lstm',
                 observation_size: int = 10,
                 action_size: int = 6,
                 hidden_size: int = 512,
                 use_auxiliary: bool = False,
                 device: str = 'auto'):
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Create network
        if network_type == 'transformer':
            self.network = TransformerPolicyNet(
                observation_size=observation_size,
                action_size=action_size,
                hidden_size=hidden_size,
                use_auxiliary=use_auxiliary
            )
        elif network_type == 'multimemory':
            self.network = MultiMemoryPolicyNet(
                observation_size=observation_size,
                action_size=action_size,
                hidden_size=hidden_size,
                use_auxiliary=use_auxiliary
            )
        else:  # lstm
            self.network = LSTMPolicyNet(
                observation_size=observation_size,
                action_size=action_size,
                hidden_size=hidden_size,
                use_auxiliary=use_auxiliary
            )
        
        self.network.to(self.device)
        self.network_type = network_type
        
    def act(self, 
            observation: np.ndarray,
            training: bool = False) -> int:
        """Select action based on observation"""
        with torch.set_grad_enabled(training):
            # Convert observation to LongTensor for embedding layer
            obs_tensor = torch.from_numpy(observation).long().to(self.device)
            
            # Add batch and sequence dimensions
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
            
            # Get action probabilities
            logits = self.network(obs_tensor)
            
            if training:
                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs.squeeze(), 1).item()
            else:
                # Take greedy action
                action = logits.argmax(dim=-1).item()
            
            return action
    
    def reset(self):
        """Reset agent state"""
        self.network.reset_state()
    
    def save(self, path: str):
        """Save agent to file"""
        self.network.save(path)
    
    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'Agent':
        """Load agent from file"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Create agent
        agent = cls(
            network_type=config.get('type', 'lstm'),
            observation_size=config.get('observation_size', 10),
            action_size=config.get('action_size', 6),
            hidden_size=config.get('hidden_size', 512),
            use_auxiliary=config.get('use_auxiliary', False),
            device=device
        )
        
        # Load weights
        if 'state_dict' in checkpoint:
            agent.network.load_state_dict(checkpoint['state_dict'])
        elif 'agent_state' in checkpoint:
            agent.network.load_state_dict(checkpoint['agent_state'])
        
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