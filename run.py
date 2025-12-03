# run.py (updated section in test_memory_model function)
def test_memory_model(args, model_path):
    """Test the trained model"""
    print(f"Testing model: {model_path}")
    
    import torch
    
    # Setup environment
    env = setup_memory_approach(args)
    
    # Determine network type from model path or args
    if 'transformer' in model_path:
        network_type = 'transformer'
    elif 'multimemory' in model_path:
        network_type = 'multimemory'
    else:
        network_type = 'lstm'
    
    # Load appropriate network
    if network_type == 'transformer':
        from simple_training import TransformerPolicyNet as PolicyNet
    elif network_type == 'multimemory':
        from simple_training import MultiMemoryPolicyNet as PolicyNet
    else:
        from simple_training import MemoryPolicyNet as PolicyNet
    
    # Network parameters
    observation_size = 10
    vocab_size = 20
    embed_dim = 512
    hidden_size = 512
    action_count = 6
    
    # Create network
    if network_type == 'transformer':
        net = PolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count,
            num_heads=8,
            num_layers=3
        )
    elif network_type == 'multimemory':
        net = PolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count,
            transformer_heads=8,
            transformer_layers=3,
            cache_size=50
        )
    else:
        net = PolicyNet(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            observation_size=observation_size,
            hidden_size=hidden_size,
            action_count=action_count
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)
    
    # Rest of the testing code remains the same...