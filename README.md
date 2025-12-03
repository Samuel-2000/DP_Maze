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