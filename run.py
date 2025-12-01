#!/usr/bin/env python3
"""
Runner for Memory Maze RL experiments
Usage:
    python run.py --mode train  # Train LSTM policy
    python run.py --mode test --model policy_epoch_10000.pt  # Test LSTM
"""

import argparse
import os
import sys
import cv2
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Memory Maze RL Runner")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both",
                       help="Train, test, or both (train then test)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (for test mode)")
    parser.add_argument("--train-epochs", type=int, default=10000,
                       help="Training epochs")
    parser.add_argument("--test-episodes", type=int, default=10,
                       help="Number of test episodes")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization during testing")
    parser.add_argument("--max-age", type=int, default=50,
                       help="Maximum age/steps per episode")
    parser.add_argument("--grid-size", type=int, default=11,
                       help="Maze grid size")
    parser.add_argument("--food-count", type=int, default=4,
                       help="Number of food sources")
    return parser.parse_args()

def setup_memory_approach(args):
    """Setup for LSTM-based memory approach"""
    # Add memory_maze directory to path
    memory_maze_dir = Path(__file__).parent / "memory_maze"
    sys.path.insert(0, str(memory_maze_dir))
    
    # Import the GridMazeWorld from memory_maze.py file
    from memory_maze import GridMazeWorld
    
    # Calculate obstacle count (25% of empty cells)
    obstacle_fraction = 0.25
    obstacle_count = int((args.grid_size - 2) * (args.grid_size - 2) * obstacle_fraction)
    
    env = GridMazeWorld(
        max_age=args.max_age,
        grid_size=args.grid_size,
        obstacle_count=obstacle_count,
        food_source_count=args.food_count,
        food_energy=10,
        initial_energy=30,
    )
    
    return env

def load_memory_policy():
    """Load the MemoryPolicyNet from simple_training.py"""
    memory_maze_dir = Path(__file__).parent / "memory_maze"
    
    # We need to import MemoryPolicyNet from simple_training.py
    # Since it's not a package, we'll use importlib
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "simple_training", 
        memory_maze_dir / "simple_training.py"
    )
    simple_training = importlib.util.module_from_spec(spec)
    sys.modules["simple_training"] = simple_training
    spec.loader.exec_module(simple_training)
    
    # Network parameters (must match training)
    observation_size = 10
    vocab_size = 20
    embed_dim = 768
    hidden_size = 768
    action_count = 8
    
    from simple_training import MemoryPolicyNet
    net = MemoryPolicyNet(
        vocab_size, embed_dim, observation_size,
        hidden_size, action_count
    )
    
    return net

def train_memory_model(args):
    """Train the LSTM policy network"""
    print("Training memory-based LSTM policy...")
    
    project_root = Path(__file__).parent
    
    cmd = [
        sys.executable, str(project_root / "memory_maze" / "simple_training.py"),
        "--epochs", str(args.train_epochs),
        "--batch-size", "64",
        "--learning-rate", "0.0005",
        "--max-age", str(args.max_age),
        "--env-group-size", "1",
        "--save-interval", "1000"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    import subprocess
    subprocess.run(cmd, cwd=project_root)

def test_memory_model(args, model_path):
    """Test the trained LSTM model"""
    print(f"Testing memory model: {model_path}")
    
    import torch
    
    # Setup environment
    env = setup_memory_approach(args)
    
    # Load network
    net = load_memory_policy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)
    
    # Test episodes
    total_reward = 0
    episodes = args.test_episodes
    
    for ep in range(episodes):
        # Reset environment and get initial observation
        initial_obs = env.reset()[0]
        
        # Reset network state
        net.reset_state()
        
        # Prepare initial observation tensor
        observations = torch.tensor([initial_obs], dtype=torch.long).to(device).unsqueeze(1)
        
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                # Get action from network
                logits = net(observations).squeeze(1)
                action = logits.argmax(dim=-1).item()
            
            # Take step in environment
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Visualize if requested
            if args.visualize:
                img = env.render(mode='')
                cv2.imshow('Memory Maze', img)
                key = cv2.waitKey(50)
                if key == ord('q'):
                    print("Testing interrupted by user")
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    # Pause on spacebar
                    while True:
                        key = cv2.waitKey(0)
                        if key == ord(' '):
                            break
                        elif key == ord('q'):
                            cv2.destroyAllWindows()
                            return
            
            # Prepare observation for next step
            observations = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
        
        total_reward += episode_reward
        print(f"Episode {ep+1}/{episodes}: Reward = {episode_reward:.2f}, Steps = {env.age}")
    
    avg_reward = total_reward / episodes
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
    
    if args.visualize:
        cv2.destroyAllWindows()

def quick_demo(model_path, episodes=3, max_steps=100):
    """Quick demonstration of trained agent"""
    print(f"\nQuick demo of trained agent ({episodes} episodes)...")
    
    import torch
    
    # Setup with default parameters
    args = argparse.Namespace(
        grid_size=11,
        max_age=max_steps,
        food_count=4,
        visualize=True
    )
    
    env = setup_memory_approach(args)
    net = load_memory_policy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)
    
    for ep in range(episodes):
        print(f"\n--- Episode {ep+1} ---")
        initial_obs = env.reset()[0]
        net.reset_state()
        observations = torch.tensor([initial_obs], dtype=torch.long).to(device).unsqueeze(1)
        
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            with torch.no_grad():
                logits = net(observations).squeeze(1)
                action = logits.argmax(dim=-1).item()
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Show visualization
            img = env.render(mode='')
            cv2.imshow('Memory Maze - Demo', img)
            key = cv2.waitKey(100)  # 100ms delay
            
            if key == ord('q'):
                break
            
            observations = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
            step += 1
            
            if done:
                break
        
        print(f"Episode reward: {episode_reward:.2f}, Steps: {step}")
    
    cv2.destroyAllWindows()

def main():
    args = parse_args()
    
    # Determine default model path
    if args.model is None:
        model_path = str(Path(__file__).parent / "memory_maze" / f"policy_epoch_{args.train_epochs:06d}.pt")
    else:
        model_path = args.model
    
    # Check if model exists
    model_exists = os.path.exists(model_path)
    
    # Execute based on mode
    if args.mode in ["train", "both"] and not model_exists:
        print(f"Model not found at {model_path}. Training new model...")
        train_memory_model(args)
        model_exists = os.path.exists(model_path)
    
    if args.mode in ["test", "both"]:
        if not model_exists:
            print(f"Error: Model not found at {model_path}")
            print("Available model files:")
            model_dir = Path(__file__).parent / "memory_maze"
            if model_dir.exists():
                for f in model_dir.glob("*.pt"):
                    print(f"  - {f.name}")
            return
        
        if args.visualize and args.test_episodes > 0:
            test_memory_model(args, model_path)
        elif args.test_episodes > 0:
            test_memory_model(args, model_path)
        else:
            # Quick demo mode
            quick_demo(model_path)

if __name__ == "__main__":
    main()