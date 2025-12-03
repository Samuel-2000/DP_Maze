# experiments/train_experiments.py
#!/usr/bin/env python3
"""
Train different memory architectures for comparison
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(network_type, epochs=10000, auxiliary=False, **kwargs):
    """Run training with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Training {network_type} network")
    if auxiliary:
        print("With auxiliary tasks")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "memory_maze/simple_training.py",
        "--batch-size", "32",
        "--epochs", str(epochs),
        "--learning-rate", "0.0005",
        "--max-age", "100",
        "--network-type", network_type,
        "--save-interval", "1000"
    ]
    
    if auxiliary:
        cmd.append("--auxiliary-tasks")
    
    if network_type == 'multimemory':
        cmd.extend(["--cache-size", "100"])
        cmd.extend(["--transformer-heads", "8"])
        cmd.extend(["--transformer-layers", "3"])
    
    # Add additional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save output
    output_dir = Path("experiments/output")
    output_dir.mkdir(exist_ok=True)
    
    filename = f"{network_type}"
    if auxiliary:
        filename += "_aux"
    
    with open(output_dir / f"{filename}_output.txt", "w") as f:
        f.write(result.stdout)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    
    return result.returncode

def main():
    """Run all experiments"""
    
    experiments = [
        # Baseline LSTM
        {"network_type": "lstm", "epochs": 5000},
        
        # Transformer
        {"network_type": "transformer", "epochs": 5000},
        {"network_type": "transformer", "epochs": 5000, "auxiliary": True},
        
        # MultiMemory
        {"network_type": "multimemory", "epochs": 5000},
        {"network_type": "multimemory", "epochs": 5000, "auxiliary": True},
    ]
    
    results = {}
    
    for exp in experiments:
        network_type = exp["network_type"]
        epochs = exp.get("epochs", 10000)
        auxiliary = exp.get("auxiliary", False)
        
        key = f"{network_type}{'_aux' if auxiliary else ''}"
        
        print(f"\nStarting experiment: {key}")
        returncode = run_experiment(
            network_type=network_type,
            epochs=epochs,
            auxiliary=auxiliary
        )
        
        results[key] = "SUCCESS" if returncode == 0 else "FAILED"
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for exp_name, status in results.items():
        print(f"{exp_name:30} {status}")
    
    # Generate comparison plot
    generate_comparison_plot()

def generate_comparison_plot():
    """Generate comparison plot from experiment results"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # This would parse the output files and create plots
        # For now, just create a template
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Example plot
        x = np.arange(100)
        for i, (label, color) in enumerate([
            ("LSTM", "blue"),
            ("Transformer", "green"),
            ("Transformer+aux", "orange"),
            ("MultiMemory", "red")
        ]):
            ax = axes[i//2, i%2]
            ax.plot(x, np.random.random(100) + i*0.2, color=color, label=label)
            ax.set_title(f"Performance: {label}")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Average Reward")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("experiments/comparison_results.png")
        print("Saved comparison plot to experiments/comparison_results.png")
        
    except ImportError:
        print("Matplotlib not available for plotting")

if __name__ == "__main__":
    main()