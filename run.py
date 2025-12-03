#!/usr/bin/env python3
"""
Unified Runner for Memory Maze RL Experiments
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.trainer import Trainer
from src.evaluation.benchmark import Benchmark
from src.evaluation.visualization import Visualizer
from src.core.utils import setup_logging, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory Maze RL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train --config configs/transformer.yaml
  python run.py train --network-type transformer --auxiliary-tasks
  python run.py test --model models/transformer_best.pt
  python run.py benchmark --benchmark-episodes 50
  python run.py visualize --model models/transformer_best.pt --save-video
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml",
                            help="Path to config file")
    train_parser.add_argument("--network-type", choices=["lstm", "transformer", "multimemory"],
                            default="lstm", help="Network architecture")
    train_parser.add_argument("--auxiliary-tasks", action="store_true",
                            help="Use auxiliary tasks")
    train_parser.add_argument("--epochs", type=int, default=10000,
                            help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.0005,
                            help="Learning rate")
    train_parser.add_argument("--save-dir", type=str, default="models",
                            help="Directory to save models")
    train_parser.add_argument("--experiment-name", type=str, default=None,
                            help="Experiment name for logging")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model", type=str, required=True,
                           help="Path to trained model")
    test_parser.add_argument("--episodes", type=int, default=10,
                           help="Number of test episodes")
    test_parser.add_argument("--visualize", action="store_true",
                           help="Show visualization")
    test_parser.add_argument("--save-video", action="store_true",
                           help="Save test video")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    bench_parser.add_argument("--models-dir", type=str, default="models",
                            help="Directory containing models")
    bench_parser.add_argument("--benchmark-episodes", type=int, default=20,
                            help="Episodes per model")
    bench_parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                            help="Output directory for results")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model performance")
    viz_parser.add_argument("--model", type=str, required=True,
                          help="Path to model")
    viz_parser.add_argument("--episodes", type=int, default=3,
                          help="Number of episodes to visualize")
    viz_parser.add_argument("--save-video", action="store_true",
                          help="Save visualization as video")
    viz_parser.add_argument("--save-gif", action="store_true",
                          help="Save as GIF")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare architectures")
    compare_parser.add_argument("--architectures", nargs="+",
                              default=["lstm", "transformer", "multimemory"],
                              help="Architectures to compare")
    compare_parser.add_argument("--epochs", type=int, default=5000,
                              help="Training epochs per architecture")
    compare_parser.add_argument("--trials", type=int, default=3,
                              help="Number of trials per architecture")
    compare_parser.add_argument("--output-dir", type=str, default="results/comparisons",
                              help="Output directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    
    if args.command == "train":
        # Load config
        config = load_config(args.config)
        
        # Override config with command line args
        if args.network_type:
            config["model"]["type"] = args.network_type
        if args.auxiliary_tasks:
            config["training"]["auxiliary_tasks"] = True
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.lr:
            config["training"]["learning_rate"] = args.lr
        
        # Create trainer and train
        trainer = Trainer(config, args.experiment_name)
        trainer.train()
        
    elif args.command == "test":
        from src.core.agent import Agent
        
        agent = Agent.load(args.model)
        test_results = agent.test(
            episodes=args.episodes,
            visualize=args.visualize,
            save_video=args.save_video
        )
        
        print(f"\nTest Results:")
        print(f"Average Reward: {test_results['avg_reward']:.2f}")
        print(f"Success Rate: {test_results['success_rate']:.1f}%")
        print(f"Average Steps: {test_results['avg_steps']:.1f}")
        
    elif args.command == "benchmark":
        benchmark = Benchmark(args.models_dir, args.output_dir)
        results = benchmark.run(args.benchmark_episodes)
        benchmark.save_results(results)
        benchmark.plot_results(results)
        
    elif args.command == "visualize":
        visualizer = Visualizer(args.model)
        visualizer.run(
            episodes=args.episodes,
            save_video=args.save_video,
            save_gif=args.save_gif
        )
        
    elif args.command == "compare":
        from experiments.compare import run_comparison
        
        run_comparison(
            architectures=args.architectures,
            epochs=args.epochs,
            trials=args.trials,
            output_dir=args.output_dir
        )
        
    else:
        print("Please specify a command. Use --help for usage information.")


if __name__ == "__main__":
    main()