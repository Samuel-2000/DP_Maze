# src/training/trainer.py - UPDATED VERSION (fixing network creation)
"""
Training module with optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

from src.core.environment import GridMazeWorld
from src.core.agent import Agent
from src.core.utils import setup_logging, seed_everything
from .losses import PolicyLoss, AuxiliaryLoss
from .optimizers import GradientClipper, LearningRateScheduler


class Trainer:
    """Main trainer class"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 experiment_name: Optional[str] = None,
                 use_wandb: bool = False):
        
        self.config = config
        self.experiment_name = experiment_name or f"exp_{np.random.randint(10000)}"
        self.use_wandb = use_wandb
        
        # Setup
        self.logger = setup_logging(self.experiment_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Set seed
        seed_everything(config.get('seed', 42))
        
        # Create environment
        self.env = self._create_environment()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        
        # Get training config
        training_config = self.config['training']
        
        # Create loss functions
        self.policy_loss_fn = PolicyLoss(
            gamma=training_config.get('gamma', 0.97),
            entropy_coef=training_config.get('entropy_coef', 0.01),
            normalize_advantages=True
        )
        
        # Only create auxiliary loss if configured
        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(
                energy_coef=0.1,
                obs_coef=0.05,
                obs_prediction_type='classification'
            )
        else:
            self.aux_loss_fn = None
            
        self.gradient_clipper = GradientClipper(
            max_norm=training_config.get('max_grad_norm', 1.0)
        )
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            mode='cosine',
            lr_start=training_config.get('learning_rate', 0.0005),
            lr_min=1e-6
        )
        
        # Metrics
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'test_rewards': [],
            'best_reward': -np.inf
        }
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="maze-rl",
                name=self.experiment_name,
                config=config
            )
    
    def _create_environment(self) -> GridMazeWorld:
        """Create training environment"""
        env_config = self.config['environment']
        return GridMazeWorld(
            grid_size=env_config.get('grid_size', 11),
            max_steps=env_config.get('max_steps', 100),
            obstacle_fraction=env_config.get('obstacle_fraction', 0.25),
            n_food_sources=env_config.get('n_food_sources', 4),
            food_energy=env_config.get('food_energy', 10.0),
            initial_energy=env_config.get('initial_energy', 30.0)
        )
    
    def _create_agent(self) -> Agent:
        """Create agent with specified network"""
        model_config = self.config['model']
        
        agent = Agent(
            network_type=model_config['type'],
            observation_size=10,  # Fixed observation size
            action_size=self.env.action_space.n,
            hidden_size=model_config.get('hidden_size', 512),
            use_auxiliary=model_config.get('use_auxiliary', False),
            device=self.device
        )
        
        # Load pretrained if specified
        if 'pretrained_path' in model_config:
            agent.load(model_config['pretrained_path'])
        
        return agent
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']
        
        optimizer_type = training_config.get('optimizer', 'adam')
        lr = training_config.get('learning_rate', 0.0005)
        weight_decay = training_config.get('weight_decay', 0.0)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train(self):
        """Main training loop"""
        training_config = self.config['training']
        epochs = training_config.get('epochs', 10000)
        batch_size = training_config.get('batch_size', 32)
        save_interval = training_config.get('save_interval', 1000)
        test_interval = training_config.get('test_interval', 500)
        
        # Create progress bar
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        for epoch in pbar:
            # Training phase
            train_metrics = self._train_epoch(batch_size)
            
            # Update metrics
            self.metrics['train_rewards'].append(train_metrics['reward'])
            self.metrics['train_losses'].append(train_metrics['loss'])
            
            # Test phase
            if (epoch + 1) % test_interval == 0:
                test_metrics = self._test_epoch(episodes=10)
                self.metrics['test_rewards'].append(test_metrics['reward'])
                
                # Update best model
                if test_metrics['reward'] > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_metrics['reward']
                    self._save_model('best')
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self._save_model(f'epoch_{epoch+1:06d}')
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{train_metrics['reward']:.2f}",
                'loss': f"{train_metrics['loss']:.4f}",
                'best': f"{self.metrics['best_reward']:.2f}"
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/reward': train_metrics['reward'],
                    'train/loss': train_metrics['loss'],
                    'train/entropy': train_metrics.get('entropy', 0.0),
                    'lr': self.lr_scheduler.get_lr(),
                })
                
                if (epoch + 1) % test_interval == 0:
                    wandb.log({
                        'test/reward': test_metrics['reward'],
                        'test/success_rate': test_metrics['success_rate'],
                    })
            
            # Update learning rate
            self.lr_scheduler.step()
        
        # Save final model
        self._save_model('final')
        
        # Save training metrics
        self._save_metrics()
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, batch_size: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.agent.network.train()
        
        # Collect experiences
        experiences = self._collect_experiences(batch_size)
        
        # Compute loss
        loss, metrics = self._compute_loss(experiences)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.gradient_clipper.clip(self.agent.network.parameters())
        self.optimizer.step()
        
        # Flush cache if using multi-memory
        if hasattr(self.agent.network, 'flush_cache_buffer'):
            self.agent.network.flush_cache_buffer()
        
        return metrics
    

    def _collect_experiences(self, n_episodes: int) -> Dict[str, torch.Tensor]:
        """Collect experiences by running episodes"""
        observations = []
        actions = []
        rewards = []
        energies = []  # For auxiliary tasks
        
        for ep in range(n_episodes):
            # Reset environment and agent
            obs, info = self.env.reset()
            self.agent.reset()  # Reset network state - IMPORTANT!
            
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            episode_energies = []
            
            terminated = truncated = False
            
            while not (terminated or truncated):
                # Store observation
                episode_obs.append(obs.copy())
                
                # Get energy from observation (last element, scaled back to 0-100)
                energy_token = obs[-1]  # Value between 12-19
                # Convert token back to energy value (approx)
                energy_value = ((energy_token - 12) / 7.0) * 100.0
                episode_energies.append(energy_value)
                
                # Select action
                action = self.agent.act(obs, training=True)
                episode_actions.append(action)
                
                # Take step
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_rewards.append(reward)
            
            # Store episode
            observations.append(episode_obs)
            actions.append(episode_actions)
            rewards.append(episode_rewards)
            energies.append(episode_energies)
        
        # Reset the agent network for batch processing
        self.agent.reset()  # Reset before batch forward pass
        
        # Pad sequences
        max_len = max(len(r) for r in rewards)


        """
        def pad_sequence(seq, max_len, pad_value=0, is_observation=False, dtype=np.int32):
            padded = []
            for s in seq:
                pad_len = max_len - len(s)
                if pad_len > 0:
                    if is_observation:
                        # For observations (nested lists)
                        pad_shape = (pad_len,) + np.array(s[0]).shape
                        padding = np.full(pad_shape, pad_value, dtype=dtype)
                        padded_s = np.concatenate([np.array(s, dtype=dtype), padding], axis=0)
                    else:
                        # For actions, rewards, energies (flat lists)
                        if dtype == np.int32:
                            padding = [pad_value] * pad_len
                        else:
                            padding = [pad_value] * pad_len
                        padded_s = np.array(s + padding, dtype=dtype)
                else:
                    padded_s = np.array(s, dtype=dtype)
                padded.append(padded_s)
            
            return np.array(padded)
        
        # Pad sequences with correct data types
        obs_padded = pad_sequence(observations, max_len, pad_value=0, is_observation=True, dtype=np.int32)
        act_padded = pad_sequence(actions, max_len, pad_value=0, is_observation=False, dtype=np.int32)
        rew_padded = pad_sequence(rewards, max_len, pad_value=0.0, is_observation=False, dtype=np.float32)
        energy_padded = pad_sequence(energies, max_len, pad_value=0.0, is_observation=False, dtype=np.float32)
        
        return {
            'observations': torch.tensor(obs_padded, dtype=torch.long).to(self.device),
            'actions': torch.tensor(act_padded, dtype=torch.long).to(self.device),
            'rewards': torch.tensor(rew_padded, dtype=torch.float32).to(self.device),
            'energies': torch.tensor(energy_padded, dtype=torch.float32).to(self.device)
        }
        """

        
        def pad_sequence(seq, max_len, pad_value=0, is_observation=False):
            """Pad a list of sequences to the same length"""
            padded = []
            for s in seq:
                pad_len = max_len - len(s)
                if pad_len > 0:
                    if is_observation:
                        # For observations (nested lists)
                        pad_shape = (pad_len,) + np.array(s[0]).shape
                        padding = np.full(pad_shape, pad_value, dtype=np.int32)
                        padded_s = np.concatenate([np.array(s), padding], axis=0)
                    else:
                        # For actions, rewards, energies (flat lists)
                        padding = [pad_value] * pad_len
                        padded_s = np.array(s + padding)
                else:
                    padded_s = np.array(s)
                padded.append(padded_s)
            
            return np.array(padded)
        
        # Pad sequences
        obs_padded = pad_sequence(observations, max_len, pad_value=0, is_observation=True)
        act_padded = pad_sequence(actions, max_len, pad_value=0)
        rew_padded = pad_sequence(rewards, max_len, pad_value=0.0)
        energy_padded = pad_sequence(energies, max_len, pad_value=0.0)
        
        return {
            'observations': torch.tensor(obs_padded, dtype=torch.long).to(self.device),
            'actions': torch.tensor(act_padded, dtype=torch.long).to(self.device),
            'rewards': torch.tensor(rew_padded, dtype=torch.float32).to(self.device),
            'energies': torch.tensor(energy_padded, dtype=torch.float32).to(self.device)
        }
    
    def _compute_loss(self, 
                    experiences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute policy loss with auxiliary losses"""
        obs = experiences['observations']
        actions = experiences['actions']
        rewards = experiences['rewards']
        energies = experiences.get('energies', None)
        
        # Create mask for valid steps (where rewards are non-zero)
        mask = (rewards != 0).float()
        
        # Forward pass
        if self.aux_loss_fn and energies is not None:
            # Get outputs from network with auxiliary predictions
            outputs = self.agent.network(obs, return_auxiliary=True)
            
            if isinstance(outputs, tuple):
                if len(outputs) == 4:
                    logits, energy_pred, obs_pred, _ = outputs
                else:
                    logits, energy_pred, obs_pred = outputs
            else:
                # Network doesn't return auxiliary outputs
                logits = outputs
                energy_pred = None
                obs_pred = None
            
            # Policy loss - actions should be [B, T]
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask  # Remove the .unsqueeze(-1)
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': policy_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
            
            # Add auxiliary loss if available
            if energy_pred is not None and obs_pred is not None:
                aux_loss = self.aux_loss_fn(
                    energy_pred, energies,
                    obs_pred, obs, mask
                )
                total_loss = total_loss + aux_loss
                metrics['aux_loss'] = aux_loss.item()
                metrics['loss'] = total_loss.item()
            
        else:
            # Standard forward pass without auxiliary tasks
            logits = self.agent.network(obs)
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask  # Remove the .unsqueeze(-1)
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
        
        return total_loss, metrics
    
    def _test_epoch(self, episodes: int = 10) -> Dict[str, float]:
        """Test agent performance"""
        self.agent.network.eval()
        
        total_reward = 0.0
        success_count = 0
        episode_lengths = []
        
        with torch.no_grad():
            for _ in range(episodes):
                obs, info = self.env.reset()
                self.agent.reset()
                
                episode_reward = 0.0
                steps = 0
                terminated = truncated = False
                
                while not (terminated or truncated) and steps < self.env.max_steps:
                    action = self.agent.act(obs, training=False)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                
                total_reward += episode_reward
                episode_lengths.append(steps)
                
                # Consider episode successful if agent survives to end
                if steps == self.env.max_steps:
                    success_count += 1
        
        avg_reward = total_reward / episodes
        success_rate = success_count / episodes * 100
        avg_length = np.mean(episode_lengths)
        
        return {
            'reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length
        }
    
    def _save_model(self, name: str):
        """Save model checkpoint"""
        save_dir = Path(self.config.get('save_dir', 'models'))
        save_dir.mkdir(exist_ok=True)
        
        # Save agent - NO with torch.no_grad() here as agent.save handles it
        agent_path = save_dir / f"{self.experiment_name}_{name}.pt"
        self.agent.save(str(agent_path))
        
        # Save optimizer and scheduler
        checkpoint_path = save_dir / f"{self.experiment_name}_{name}_checkpoint.pt"
        torch.save({
            'epoch': len(self.metrics['train_rewards']),
            'agent_state': self.agent.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, str(checkpoint_path))
        
        self.logger.info(f"Saved model to {agent_path}")
    
    def _save_metrics(self):
        """Save training metrics"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_path = metrics_dir / f"{self.experiment_name}_metrics.npz"
        np.savez(str(metrics_path), **self.metrics)
        
        # Plot metrics
        self._plot_metrics()
    
    def _plot_metrics(self):
        """Plot training metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training rewards
        axes[0, 0].plot(self.metrics['train_rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training losses
        axes[0, 1].plot(self.metrics['train_losses'])
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Test rewards
        if self.metrics['test_rewards']:
            axes[1, 0].plot(np.arange(0, len(self.metrics['train_rewards']), 
                                    len(self.metrics['train_rewards']) // len(self.metrics['test_rewards'])),
                          self.metrics['test_rewards'], 'o-')
            axes[1, 0].set_title('Test Rewards')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined plot
        window = 100
        if len(self.metrics['train_rewards']) >= window:
            smooth_rewards = np.convolve(self.metrics['train_rewards'], 
                                       np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smooth_rewards, label='Smoothed')
            axes[1, 1].set_title(f'Smoothed Rewards (window={window})')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = Path('results/plots') / f"{self.experiment_name}_metrics.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved metrics plot to {plot_path}")