# test_step_by_step.py
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing step by step...")

# Test 1: Create simple observation
print("\n1. Testing observation format...")
obs = np.random.randint(0, 20, size=10, dtype=np.int32)
print(f"Single observation: shape={obs.shape}, type={type(obs)}, dtype={obs.dtype}")

# Test 2: Create batch of observations
print("\n2. Testing batch creation...")
batch_size = 4
seq_length = 5

# Create a batch of sequences
obs_batch = []
for b in range(batch_size):
    seq = []
    for t in range(seq_length):
        seq.append(np.random.randint(0, 20, size=10, dtype=np.int32))
    obs_batch.append(seq)

print(f"Batch structure: {len(obs_batch)} sequences, each with {len(obs_batch[0])} timesteps")
print(f"Each timestep shape: {obs_batch[0][0].shape}")

# Test 3: Convert to tensor properly
print("\n3. Converting to tensor...")

def pad_observations(batch, max_len, pad_value=0):
    """Properly pad a batch of observation sequences"""
    batch_size = len(batch)
    obs_dim = len(batch[0][0])  # Should be 10
    
    # Create empty tensor
    padded = np.full((batch_size, max_len, obs_dim), pad_value, dtype=np.int32)
    
    for i, seq in enumerate(batch):
        seq_len = min(len(seq), max_len)
        for t in range(seq_len):
            padded[i, t] = seq[t]
    
    return torch.tensor(padded, dtype=torch.long)

max_len = max(len(seq) for seq in obs_batch)
obs_tensor = pad_observations(obs_batch, max_len)
print(f"Padded tensor shape: {obs_tensor.shape}")
print(f"Tensor dtype: {obs_tensor.dtype}")

print("\n✓ All tests passed!")