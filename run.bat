@echo off
echo ==========================================
echo MASTER THESIS - Memory Maze RL Experiments
echo ==========================================
echo One-click setup, train, test, and benchmark
echo.

cd /d "%~dp0"

REM ==========================================
REM 1. SETUP
REM ==========================================
echo [1/5] SETUP: Installing dependencies...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Install packages
echo Installing required packages...
pip install --upgrade pip >nul 2>&1
pip install numba gym scikit-image opencv-python stable_baselines3 torch tqdm matplotlib pandas seaborn >nul 2>&1

REM Create directories
mkdir models 2>nul
mkdir experiments 2>nul
mkdir experiments\output 2>nul
mkdir benchmark 2>nul
mkdir benchmark\results 2>nul

echo Setup complete! ✓
echo.

REM ==========================================
REM 2. TRAIN ALL MODELS
REM ==========================================
echo [2/5] TRAINING: Training all memory architectures...
echo This may take 15-30 minutes...
echo.

set TRAIN_CMD=python memory_maze\simple_training.py --batch-size 32 --learning-rate 0.0005 --max-age 100 --save-interval 1000 --epochs 2000

echo Training LSTM baseline...
%TRAIN_CMD% --network-type lstm > experiments\output\lstm_training.log 2>&1

echo Training Transformer...
%TRAIN_CMD% --network-type transformer --transformer-heads 8 --transformer-layers 3 >> experiments\output\transformer_training.log 2>&1

echo Training MultiMemory...
%TRAIN_CMD% --network-type multimemory --transformer-heads 8 --transformer-layers 3 --cache-size 50 >> experiments\output\multimemory_training.log 2>&1

echo Training complete! ✓
echo Models saved in: models\ folder
echo.

REM ==========================================
REM 3. TEST MODELS
REM ==========================================
echo [3/5] TESTING: Running test episodes...
echo.

REM Find latest models
for /f "delims=" %%i in ('dir models\policy_lstm_*.pt /b /od 2^>nul') do set LATEST_LSTM=%%i
for /f "delims=" %%i in ('dir models\policy_transformer_*.pt /b /od 2^>nul') do set LATEST_TRANSFORMER=%%i
for /f "delims=" %%i in ('dir models\policy_multimemory_*.pt /b /od 2^>nul') do set LATEST_MULTIMEMORY=%%i

if not defined LATEST_LSTM (
    echo ERROR: No LSTM model found!
    goto :RESULTS
)

echo Testing LSTM: %LATEST_LSTM%
python run.py --mode test --model "models\%LATEST_LSTM%" --test-episodes 5 --max-age 100 --visualize > experiments\output\lstm_test.log 2>&1

if defined LATEST_TRANSFORMER (
    echo Testing Transformer: %LATEST_TRANSFORMER%
    python run.py --mode test --model "models\%LATEST_TRANSFORMER%" --test-episodes 5 --max-age 100 > experiments\output\transformer_test.log 2>&1
)

if defined LATEST_MULTIMEMORY (
    echo Testing MultiMemory: %LATEST_MULTIMEMORY%
    python run.py --mode test --model "models\%LATEST_MULTIMEMORY%" --test-episodes 5 --max-age 100 > experiments\output\multimemory_test.log 2>&1
)

echo Testing complete! ✓
echo.

REM ==========================================
REM 4. BENCHMARK
REM ==========================================
echo [4/5] BENCHMARK: Running performance benchmarks...
echo.

REM Create benchmark script if it doesn't exist
if not exist "benchmark.py" (
    echo Creating benchmark script...
    copy con benchmark.py >nul 2>&1 <<EOF
import torch, pandas as pd, numpy as np, glob, os, sys
from pathlib import Path
sys.path.insert(0, '.')
from memory_maze.simple_training import MemoryPolicyNet, TransformerPolicyNet, MultiMemoryPolicyNet
from memory_maze import GridMazeWorld

def benchmark_model(model_path, episodes=20, max_age=100):
    model_name = Path(model_path).name
    
    observation_size, vocab_size, embed_dim, hidden_size, action_count = 10, 20, 512, 512, 6
    
    if 'transformer' in model_name:
        net = TransformerPolicyNet(vocab_size, embed_dim, observation_size, hidden_size, action_count, 8, 3)
    elif 'multimemory' in model_name:
        net = MultiMemoryPolicyNet(vocab_size, embed_dim, observation_size, hidden_size, action_count, 8, 3, 50)
    else:
        net = MemoryPolicyNet(vocab_size, embed_dim, observation_size, hidden_size, action_count)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval().to(device)
    
    grid_size, obstacle_fraction = 11, 0.25
    obstacle_count = int((grid_size - 2) * (grid_size - 2) * obstacle_fraction)
    env = GridMazeWorld(max_age=max_age, grid_size=grid_size, obstacle_count=obstacle_count, food_source_count=4, food_energy=10, initial_energy=30)
    
    results = []
    for ep in range(episodes):
        net.reset_state()
        obs = env.reset()[0]
        obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
        episode_reward, steps = 0, 0
        
        for _ in range(max_age):
            with torch.no_grad():
                logits = net(obs_tensor).squeeze(1)
                action = logits.argmax(dim=-1).item()
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            obs_tensor = torch.tensor([obs], dtype=torch.long).to(device).unsqueeze(1)
            if done: break
        
        results.append({'reward': episode_reward, 'steps': steps, 'survived': not done})
    
    df = pd.DataFrame(results)
    return {
        'model': model_name,
        'avg_reward': df['reward'].mean(),
        'std_reward': df['reward'].std(),
        'avg_steps': df['steps'].mean(),
        'survival_rate': df['survived'].mean() * 100
    }

if __name__ == "__main__":
    models = glob.glob('models/*.pt')
    if not models:
        print("No models found!")
        exit()
    
    print("Running benchmarks for each model (20 episodes each)...")
    all_results = []
    
    for model_path in models[:3]:  # Test up to 3 latest models
        print(f"Testing: {os.path.basename(model_path)}")
        results = benchmark_model(model_path)
        all_results.append(results)
    
    df_results = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(df_results.to_string(index=False))
    print("\n✓ Benchmark complete!")
    
    df_results.to_csv('benchmark/results/benchmark_summary.csv', index=False)
    print("Results saved to: benchmark/results/benchmark_summary.csv")
EOF
)

echo Running benchmark (20 episodes per model)...
python benchmark.py > experiments\output\benchmark.log 2>&1

echo Benchmark complete! ✓
echo.

REM ==========================================
REM 5. RESULTS
REM ==========================================
:RESULTS
echo [5/5] RESULTS: Summary of experiments...
echo.
echo ==========================================
echo EXPERIMENT COMPLETE!
echo ==========================================
echo.
echo OUTPUT FILES:
echo - Models:           models\*.pt
echo - Training logs:    experiments\output\*.log
echo - Benchmark results: benchmark\results\benchmark_summary.csv
echo.
echo NEXT STEPS:
echo 1. Check benchmark results in the console above
echo 2. View detailed logs in experiments\output\ folder
echo 3. For visualization: python run.py --mode test --model models\YOUR_MODEL.pt --visualize
echo.
echo ==========================================
pause