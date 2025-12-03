#!/usr/bin/env python3
"""
Simple runner to test the complete setup
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.environment import GridMazeWorld
from src.core.agent import Agent
from src.training.losses import PolicyLoss
from src.training.optimizers import GradientClipper
from src.evaluation.benchmark import Benchmark
from src.evaluation.visualization import Visualizer
from src.core.utils import setup_logging, seed_everything


def test_all_components():
    """Test all components"""
    print("="*60)
    print("Testing Maze RL Components")
    print("="*60)
    
    # 1. Test environment
    print("\n1. Testing Environment...")
    try:
        env = GridMazeWorld(grid_size=5, max_steps=10)
        obs, info = env.reset()
        print(f"✓ Environment created: obs shape={obs.shape}, action space={env.action_space.n}")
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False
    
    # 2. Test agent
    print("\n2. Testing Agent...")
    try:
        agent = Agent(
            network_type="lstm",
            observation_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            hidden_size=64
        )
        print(f"✓ Agent created: network_type={agent.network_type}")
        print(f"  Parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    except Exception as e:
        print(f"✗ Agent error: {e}")
        return False
    
    # 3. Test interaction
    print("\n3. Testing Agent-Environment Interaction...")
    try:
        obs, info = env.reset()
        agent.reset()
        
        action = agent.act(obs, training=False)
        print(f"✓ Action selected: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step taken: reward={reward:.3f}, done={terminated or truncated}")
    except Exception as e:
        print(f"✗ Interaction error: {e}")
        return False
    
    # 4. Test loss functions
    print("\n4. Testing Loss Functions...")
    try:
        import torch
        policy_loss = PolicyLoss(gamma=0.97)
        
        # Create dummy data
        B, T, A = 2, 5, env.action_space.n
        logits = torch.randn(B, T, A)
        actions = torch.randint(0, A, (B, T))
        rewards = torch.randn(B, T)
        
        loss, entropy = policy_loss(logits, actions, rewards)
        print(f"✓ Loss computed: loss={loss.item():.4f}, entropy={entropy.item():.4f}")
    except Exception as e:
        print(f"✗ Loss function error: {e}")
        return False
    
    # 5. Test gradient clipping
    print("\n5. Testing Gradient Clipping...")
    try:
        clipper = GradientClipper(max_norm=1.0)
        
        # Create dummy parameters with gradients
        params = [torch.randn(10, 10, requires_grad=True) for _ in range(3)]
        
        # Simulate backward pass
        dummy_loss = sum(p.sum() for p in params)
        dummy_loss.backward()
        
        # Clip gradients
        clipper.clip(params)
        grad_norm = clipper.get_grad_norm(params)
        print(f"✓ Gradient clipping: norm={grad_norm:.4f}")
    except Exception as e:
        print(f"✗ Gradient clipping error: {e}")
        return False
    
    # 6. Test utilities
    print("\n6. Testing Utilities...")
    try:
        logger = setup_logging("test")
        seed_everything(42)
        print("✓ Utilities: logging and seeding")
    except Exception as e:
        print(f"✗ Utilities error: {e}")
        return False
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    
    return True


def quick_demo():
    """Run a quick demo"""
    print("\n" + "="*60)
    print("Running Quick Demo")
    print("="*60)
    
    # Create environment
    env = GridMazeWorld(
        grid_size=11,
        max_steps=50,
        obstacle_fraction=0.2,
        n_food_sources=3
    )
    
    # Create agent
    agent = Agent(
        network_type="lstm",
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_size=128
    )
    
    # Run a few episodes
    print("\nRunning 3 episodes...")
    for episode in range(3):
        obs, info = env.reset()
        agent.reset()
        
        total_reward = 0
        steps = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Steps={steps}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")
    args = parser.parse_args()
    
    if args.test:
        success = test_all_components()
        if not success:
            print("\nSome tests failed. Please check the errors above.")
    elif args.demo:
        quick_demo()
    else:
        # Default: test then demo
        print("Running tests followed by demo...\n")
        if test_all_components():
            quick_demo()