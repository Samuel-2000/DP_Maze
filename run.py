#!/usr/bin/env python3
"""
Unified Runner for Memory Maze RL Experiments

Usage:
    # Train LSTM baseline
    python run.py --mode train --network-type lstm
    
    # Train Transformer with auxiliary tasks
    python run.py --mode train --network-type transformer --auxiliary-tasks
    
    # Test a specific model
    python run.py --mode test --model models/policy_lstm_epoch_010000.pt
    
    # Benchmark all available models
    python run.py --mode benchmark
    
    # Quick demo with visualization
    python run.py --mode demo --model models/policy_lstm_epoch_010000.pt
"""

import argparse
import os
import sys
import cv2
import torch
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Memory Maze RL Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train LSTM baseline:        python run.py --mode train --network-type lstm
  Train Transformer:          python run.py --mode train --network-type transformer
  Train MultiMemory:          python run.py --mode train --network-type multimemory
  Test model:                 python run.py --mode test --model models/policy_lstm_epoch_010000.pt
  Quick demo:                 python run.py --mode demo --model models/policy_lstm_epoch_010000.pt
  Benchmark all models:       python run.py --mode benchmark
  Compare architectures:      python run.py --mode compare --epochs 5000
        """
    )
    
    # Main mode
    parser.add_argument("--mode", 
                       choices=["train", "test", "demo", "benchmark", "compare", "all"],
                       default="all",
                       help="Operation mode: train, test, demo, benchmark, compare, or all")
    
    # Model parameters
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model (for test/demo mode)")
    parser.add_argument("--network-type", 
                       choices=["lstm", "transformer", "multimemory"],
                       default="lstm",
                       help="Type of memory network to use")
    parser.add_argument("--auxiliary-tasks", action="store_true",
                       help="Use auxiliary tasks for training (energy and observation prediction)")
    
    # Training parameters
    parser.add_argument("--train-epochs", type=int, default=10000,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.0005,
                       help="Learning rate")
    parser.add_argument("--save-interval", type=int, default=1000,
                       help="Save model every N epochs")
    
    # Environment parameters
    parser.add_argument("--grid-size", type=int, default=11,
                       help="Maze grid size")
    parser.add_argument("--max-age", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--food-count", type=int, default=4,
                       help="Number of food sources")
    parser.add_argument("--obstacle-fraction", type=float, default=0.25,
                       help="Fraction of grid cells that are obstacles")
    
    # Testing parameters
    parser.add_argument("--test-episodes", type=int, default=10,
                       help="Number of test episodes")
    parser.add_argument("--visualize", action="store_true",
                       help="Show visualization during testing")
    
    # Advanced network parameters
    parser.add_argument("--transformer-heads", type=int, default=8,
                       help="Number of attention heads for transformer")
    parser.add_argument("--transformer-layers", type=int, default=3,
                       help="Number of transformer layers")
    parser.add_argument("--cache-size", type=int, default=50,
                       help="Size of neural cache (for multimemory network)")
    
    # Benchmark parameters
    parser.add_argument("--benchmark-episodes", type=int, default=20,
                       help="Number of episodes per model for benchmarking")
    
    # Experimental
    parser.add_argument("--env-group-size", type=int, default=1,
                       help="Number of environment copies for training (reduces variance)")
    
    return parser.parse_args()

def setup_environment(args):
    """Setup the GridMazeWorld environment"""
    # Add memory_maze directory to path
    memory_maze_dir = Path(__file__).parent / "memory_maze"
    sys.path.insert(0, str(memory_maze_dir))
    
    from memory_maze import GridMazeWorld
    
    # Calculate obstacle count
    obstacle_count = int((args.grid_size - 2) * (args.grid_size - 2) * args.obstacle_fraction)
    
    env = GridMazeWorld(
        max_age=args.max_age,
        grid_size=args.grid_size,
        obstacle_count=obstacle_count,
        food_source_count=args.food_count,
        food_energy=10,
        initial_energy=30,
    )
    
    return env

def create_network(network_type, model_path=None, device=None):
    """Create or load a neural network based on type"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Network parameters
    observation_size = 10
    vocab_size = 20
    embed_dim = 512
    hidden_size = 512
    action_count = 6  # Updated to match Actions enum
    
    # Import appropriate network class
    try:
        from simple_training import (
            MemoryPolicyNet, 
            TransformerPolicyNet, 
            MultiMemoryPolicyNet
        )
    except ImportError:
        # Fallback: try direct import
        memory_maze_dir = Path(__file__).parent / "memory_maze"
        sys.path.insert(0, str(memory_maze_dir))
        from simple_training import (
            MemoryPolicyNet, 
            TransformerPolicyNet, 
            MultiMemoryPolicyNet
        )
    
    # Create network based on type
    if network_type == 'transformer':
        net = TransformerPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count,
            num_heads=args.transformer_heads,
            num_layers=args.transformer_layers
        )
    elif network_type == 'multimemory':
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
    else:  # lstm (default)
        net = MemoryPolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count
        )
    
    # Load weights if model path is provided
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        net.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {model_path}")
    
    net.eval()
    net.to(device)
    
    return net

def train_model(args):
    """Train a model with specified parameters"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {args.network_type.upper()} Network")
    if args.auxiliary_tasks:
        print("With Auxiliary Tasks")
    print(f"{'='*60}")
    
    # Build command for training script
    cmd = [
        sys.executable, "memory_maze/simple_training.py",
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.train_epochs),
        "--learning-rate", str(args.learning_rate),
        "--max-age", str(args.max_age),
        "--network-type", args.network_type,
        "--save-interval", str(args.save_interval),
        "--env-group-size", str(args.env_group_size),
    ]
    
    # Add optional arguments
    if args.auxiliary_tasks:
        cmd.append("--auxiliary-tasks")
    
    if args.network_type == 'transformer':
        cmd.extend(["--transformer-heads", str(args.transformer_heads)])
        cmd.extend(["--transformer-layers", str(args.transformer_layers)])
    
    if args.network_type == 'multimemory':
        cmd.extend(["--cache-size", str(args.cache_size)])
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Training for {args.train_epochs} epochs...")
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed successfully!")
        
        # Return path to the latest model
        model_dir = Path("models")
        model_files = list(model_dir.glob(f"policy_{args.network_type}_*.pt"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            return str(latest_model)
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        return None

def test_model(args, model_path):
    """Test a trained model"""
    print(f"\n{'='*60}")
    print(f"TESTING: {Path(model_path).name}")
    print(f"{'='*60}")
    
    # Setup
    env = setup_environment(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine network type from model name
    model_name = Path(model_path).name
    if 'transformer' in model_name:
        network_type = 'transformer'
    elif 'multimemory' in model_name:
        network_type = 'multimemory'
    else:
        network_type = 'lstm'
    
    # Load network
    net = create_network(network_type, model_path, device)
    
    # Test episodes
    results = []
    total_reward = 0
    
    for ep in tqdm(range(args.test_episodes), desc="Testing"):
        # Reset environment and network
        obs = env.reset()[0]
        net.reset_state()
        
        # Prepare observation tensor
        obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
        
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < args.max_age:
            # Get action from network
            with torch.no_grad():
                logits = net(obs_tensor).squeeze(1)
                action = logits.argmax(dim=-1).item()
            
            # Take step in environment
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Visualize if requested
            if args.visualize:
                img = env.render(mode='')
                cv2.imshow('Memory Maze - Testing', img)
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
            obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
        
        results.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'steps': steps,
            'survived': not done
        })
        total_reward += episode_reward
    
    # Print results
    df_results = pd.DataFrame(results)
    print(f"\nTest Results ({args.test_episodes} episodes):")
    print(df_results.to_string(index=False))
    
    avg_reward = total_reward / args.test_episodes
    survival_rate = df_results['survived'].mean() * 100
    avg_steps = df_results['steps'].mean()
    
    print(f"\nSummary:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Survival Rate: {survival_rate:.1f}%")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    if args.visualize:
        cv2.destroyAllWindows()
    
    return df_results

def run_demo(args, model_path):
    """Run a quick demonstration with visualization"""
    print(f"\n{'='*60}")
    print("DEMO MODE: Running agent with visualization")
    print(f"{'='*60}")
    
    # Setup
    env = setup_environment(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine network type from model name
    model_name = Path(model_path).name
    if 'transformer' in model_name:
        network_type = 'transformer'
    elif 'multimemory' in model_name:
        network_type = 'multimemory'
    else:
        network_type = 'lstm'
    
    # Load network
    net = create_network(network_type, model_path, device)
    
    print("Controls:")
    print("  [SPACE] - Pause/Resume")
    print("  [Q]     - Quit demo")
    print("  [N]     - Next episode")
    print("\nStarting demo...")
    
    episode = 0
    while episode < 3:  # Run 3 episodes by default
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset environment and network
        obs = env.reset()[0]
        net.reset_state()
        
        # Prepare observation tensor
        obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
        
        episode_reward = 0
        steps = 0
        done = False
        paused = False
        
        while not done and steps < args.max_age:
            if not paused:
                # Get action from network
                with torch.no_grad():
                    logits = net(obs_tensor).squeeze(1)
                    action = logits.argmax(dim=-1).item()
                
                # Take step in environment
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Prepare observation for next step
                obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
            
            # Show visualization
            img = env.render(mode='')
            cv2.imshow('Memory Maze - Demo', img)
            
            # Handle keyboard input
            key = cv2.waitKey(100 if not paused else 0)
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("Demo ended by user")
                return
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('n'):
                break
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return
        
        print(f"Episode reward: {episode_reward:.2f}, Steps: {steps}")
        episode += 1
    
    cv2.destroyAllWindows()
    print("\n✓ Demo completed!")

def benchmark_models(args):
    """Run benchmark on all available models"""
    print(f"\n{'='*60}")
    print("BENCHMARK: Comparing all available models")
    print(f"{'='*60}")
    
    # Find all model files
    model_dir = Path("models")
    if not model_dir.exists():
        print("✗ No models directory found. Train models first.")
        return
    
    model_files = list(model_dir.glob("*.pt"))
    if not model_files:
        print("✗ No model files found. Train models first.")
        return
    
    print(f"Found {len(model_files)} model(s):")
    for m in model_files:
        print(f"  - {m.name}")
    
    # Create results directory
    results_dir = Path("benchmark") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Benchmark each model
    all_results = []
    
    for model_path in tqdm(model_files, desc="Benchmarking models"):
        model_name = model_path.name
        
        # Determine network type
        if 'transformer' in model_name:
            network_type = 'transformer'
        elif 'multimemory' in model_name:
            network_type = 'multimemory'
        else:
            network_type = 'lstm'
        
        print(f"\nBenchmarking: {model_name}")
        
        # Setup for testing
        test_args = argparse.Namespace(
            grid_size=args.grid_size,
            max_age=args.max_age,
            food_count=args.food_count,
            obstacle_fraction=args.obstacle_fraction,
            test_episodes=args.benchmark_episodes,
            visualize=False
        )
        
        try:
            # Run test without visualization
            results = test_model(test_args, str(model_path))
            if results is not None:
                model_results = {
                    'model': model_name,
                    'network_type': network_type,
                    'avg_reward': results['reward'].mean(),
                    'std_reward': results['reward'].std(),
                    'avg_steps': results['steps'].mean(),
                    'survival_rate': results['survived'].mean() * 100,
                    'max_reward': results['reward'].max(),
                    'min_reward': results['reward'].min()
                }
                all_results.append(model_results)
        except Exception as e:
            print(f"✗ Error benchmarking {model_name}: {e}")
    
    # Save and display results
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        # Sort by average reward (descending)
        df_results = df_results.sort_values('avg_reward', ascending=False)
        
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(df_results.to_string(index=False))
        
        # Save to CSV
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / f"benchmark_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Save summary markdown
        md_path = results_dir / f"benchmark_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write("# Memory Maze Benchmark Results\n\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Episodes per model: {args.benchmark_episodes}\n")
            f.write(f"Max steps per episode: {args.max_age}\n\n")
            f.write("## Results\n\n")
            f.write(df_results.to_markdown(index=False))
        
        return df_results
    else:
        print("✗ No results to display")
        return None

def compare_architectures(args):
    """Compare different network architectures"""
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    
    architectures = ['lstm', 'transformer', 'multimemory']
    results = []
    
    for arch in architectures:
        print(f"\nTraining {arch} architecture...")
        
        # Update args for this architecture
        arch_args = argparse.Namespace(**vars(args))
        arch_args.network_type = arch
        arch_args.train_epochs = min(5000, args.train_epochs)  # Limit for comparison
        
        # Train model
        model_path = train_model(arch_args)
        
        if model_path:
            # Test model
            test_args = argparse.Namespace(
                grid_size=args.grid_size,
                max_age=args.max_age,
                food_count=args.food_count,
                obstacle_fraction=args.obstacle_fraction,
                test_episodes=10,
                visualize=False
            )
            
            try:
                test_results = test_model(test_args, model_path)
                if test_results is not None:
                    results.append({
                        'architecture': arch,
                        'model': Path(model_path).name,
                        'avg_reward': test_results['reward'].mean(),
                        'survival_rate': test_results['survived'].mean() * 100,
                        'avg_steps': test_results['steps'].mean()
                    })
            except Exception as e:
                print(f"✗ Error testing {arch}: {e}")
    
    # Display comparison
    if results:
        df_comparison = pd.DataFrame(results)
        df_comparison = df_comparison.sort_values('avg_reward', ascending=False)
        
        print(f"\n{'='*60}")
        print("ARCHITECTURE COMPARISON RESULTS")
        print(f"{'='*60}")
        print(df_comparison.to_string(index=False))
        
        # Save comparison
        results_dir = Path("experiments") / "output"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = results_dir / "architecture_comparison.csv"
        df_comparison.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to: {csv_path}")
        
        return df_comparison
    else:
        print("✗ No comparison results")
        return None

def run_all(args):
    """Run the complete pipeline: setup, train, test, benchmark"""
    print("\n" + "="*60)
    print("COMPLETE EXPERIMENT PIPELINE")
    print("="*60)
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    Path("experiments/output").mkdir(parents=True, exist_ok=True)
    Path("benchmark/results").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Train
    print("\n[1/4] TRAINING")
    model_path = train_model(args)
    
    if not model_path:
        print("✗ Training failed. Aborting pipeline.")
        return
    
    # Step 2: Test with visualization
    print("\n[2/4] TESTING")
    test_args = argparse.Namespace(**vars(args))
    test_args.test_episodes = 5
    test_args.visualize = True
    test_model(test_args, model_path)
    
    # Step 3: Benchmark (if multiple models exist)
    print("\n[3/4] BENCHMARKING")
    benchmark_models(args)
    
    # Step 4: Quick demo
    print("\n[4/4] DEMONSTRATION")
    run_demo(args, model_path)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  - Models:            models/")
    print("  - Training logs:     experiments/output/")
    print("  - Benchmark results: benchmark/results/")
    print("\nTo visualize: python run.py --mode demo --model models/YOUR_MODEL.pt")

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    
    # Execute based on mode
    if args.mode == "train":
        train_model(args)
    
    elif args.mode == "test":
        if args.model is None:
            # Find latest model of specified type
            model_dir = Path("models")
            model_pattern = f"policy_{args.network_type}_*.pt"
            model_files = list(model_dir.glob(model_pattern))
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                args.model = str(latest_model)
            else:
                print(f"✗ No {args.network_type} model found. Train one first.")
                return
        
        if not Path(args.model).exists():
            print(f"✗ Model not found: {args.model}")
            return
        
        test_model(args, args.model)
    
    elif args.mode == "demo":
        if args.model is None:
            # Find any model
            model_dir = Path("models")
            model_files = list(model_dir.glob("*.pt"))
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                args.model = str(latest_model)
            else:
                print("✗ No model found. Train one first.")
                return
        
        if not Path(args.model).exists():
            print(f"✗ Model not found: {args.model}")
            return
        
        run_demo(args, args.model)
    
    elif args.mode == "benchmark":
        benchmark_models(args)
    
    elif args.mode == "compare":
        compare_architectures(args)
    
    elif args.mode == "all":
        run_all(args)
    
    else:
        print(f"✗ Unknown mode: {args.mode}")
        print("Use --help for usage information.")

if __name__ == "__main__":
    main()