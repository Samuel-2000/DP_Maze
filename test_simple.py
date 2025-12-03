# test_simple.py
"""
Simple test script to verify everything works
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
print("Testing imports...")
try:
    from src.core.environment import GridMazeWorld
    from src.core.agent import Agent
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Creating minimal structure...")
    
    # Create minimal structure
    Path("src/core").mkdir(parents=True, exist_ok=True)
    Path("src/networks").mkdir(parents=True, exist_ok=True)
    Path("src/training").mkdir(parents=True, exist_ok=True)
    Path("src/evaluation").mkdir(parents=True, exist_ok=True)
    
    # Create empty __init__.py files
    for dir_path in ["src", "src/core", "src/networks", "src/training", "src/evaluation"]:
        (Path(dir_path) / "__init__.py").touch()
    
    print("Created directory structure. Please copy the provided files.")
    sys.exit(1)

# Test environment
print("\nTesting environment...")
try:
    env = GridMazeWorld(grid_size=5, max_steps=10)
    obs, info = env.reset()
    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"✗ Environment error: {e}")

# Test agent
print("\nTesting agent...")
try:
    agent = Agent(
        network_type="lstm",
        observation_size=10,
        action_size=6,
        hidden_size=64
    )
    print(f"✓ Agent created")
    print(f"  Network type: {agent.network_type}")
    print(f"  Number of parameters: {sum(p.numel() for p in agent.network.parameters())}")
except Exception as e:
    print(f"✗ Agent error: {e}")

# Test interaction
print("\nTesting agent-environment interaction...")
try:
    env = GridMazeWorld(grid_size=5, max_steps=5)
    agent = Agent(
        network_type="lstm",
        observation_size=10,
        action_size=6,
        hidden_size=64
    )
    
    obs, info = env.reset()
    agent.reset()
    
    for step in range(5):
        action = agent.act(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step}: Action={action}, Reward={reward:.3f}")
    
    print("✓ Interaction successful")
except Exception as e:
    print(f"✗ Interaction error: {e}")

print("\nTest completed!")