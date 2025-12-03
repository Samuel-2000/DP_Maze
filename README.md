# maze-rl
Experiments with RL in 2D maze environments


Advanced Memory Architectures:
    TransformerMemory: Uses multi-head attention for long-term dependencies
    NeuralCache: Content-addressable memory for storing important experiences
    MultiMemoryPolicyNet: Combines LSTM, Transformer, and cache

Auxiliary Tasks:
    Energy prediction: Predict current energy level
    Observation prediction: Predict next observation (self-supervised)

Training Improvements:
    Multiple network types via --network-type argument
    Auxiliary losses for better representations
    Better gradient clipping and optimization

Experiment Framework:
    Script to compare different architectures
    Automatic logging and plotting




# Train a model
python run.py --mode train --network-type transformer --auxiliary-tasks

# Test with visualization
python run.py --mode test --model models/policy_transformer_epoch_010000.pt --visualize

# Run benchmark on all models
python run.py --mode benchmark

# Compare different architectures
python run.py --mode compare --train-epochs 2000

# Run the complete pipeline
python run.py --mode all