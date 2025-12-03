@echo off
echo ========================================
echo Architecture Comparison Experiment
echo ========================================

setlocal enabledelayedexpansion

set EPOCHS=2000
set TRIALS=3
set ARCHITECTURES=lstm transformer multimemory

echo.
echo Running comparison experiment...
echo Epochs per trial: %EPOCHS%
echo Number of trials: %TRIALS%
echo Architectures: %ARCHITECTURES%

set OUTPUT_DIR=results\comparisons_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%
mkdir %OUTPUT_DIR%

echo.
echo Output directory: %OUTPUT_DIR%

for %%a in (%ARCHITECTURES%) do (
    echo.
    echo ========================================
    echo Testing architecture: %%a
    echo ========================================
    
    for /l %%t in (1,1,%TRIALS%) do (
        echo Trial %%t of %TRIALS%...
        python run.py train ^
            --network-type %%a ^
            --epochs %EPOCHS% ^
            --experiment-name "compare_%%a_trial%%t" ^
            --save-dir %OUTPUT_DIR%
    )
)

echo.
echo ========================================
echo Generating comparison plots...
echo ========================================

python -c "
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

architectures = ['lstm', 'transformer', 'multimemory']
data = []

for arch in architectures:
    for trial in range(1, 4):
        path = Path('results/comparisons') / f'compare_{arch}_trial{trial}_metrics.npz'
        if path.exists():
            metrics = np.load(path)
            if 'train_rewards' in metrics:
                rewards = metrics['train_rewards']
                for epoch, reward in enumerate(rewards):
                    data.append({
                        'architecture': arch,
                        'trial': trial,
                        'epoch': epoch,
                        'reward': reward
                    })

if data:
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot individual trials
    for arch in architectures:
        arch_data = df[df['architecture'] == arch]
        if not arch_data.empty:
            for trial in arch_data['trial'].unique():
                trial_data = arch_data[arch_data['trial'] == trial]
                axes[0,0].plot(trial_data['epoch'], trial_data['reward'], 
                             label=f'{arch} (trial {trial})', alpha=0.6)
    
    axes[0,0].set_title('Individual Trials')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Reward')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot averaged
    for arch in architectures:
        arch_data = df[df['architecture'] == arch]
        if not arch_data.empty:
            avg_rewards = arch_data.groupby('epoch')['reward'].mean()
            std_rewards = arch_data.groupby('epoch')['reward'].std()
            axes[0,1].plot(avg_rewards.index, avg_rewards, label=arch, linewidth=2)
            axes[0,1].fill_between(avg_rewards.index, 
                                 avg_rewards - std_rewards,
                                 avg_rewards + std_rewards, alpha=0.3)
    
    axes[0,1].set_title('Averaged (mean ± std)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Reward')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_performance = []
    for arch in architectures:
        arch_data = df[df['architecture'] == arch]
        if not arch_data.empty:
            final_rewards = arch_data.groupby('trial')['reward'].last()
            for trial, reward in final_rewards.items():
                final_performance.append({'architecture': arch, 'trial': trial, 'final_reward': reward})
    
    if final_performance:
        final_df = pd.DataFrame(final_performance)
        axes[1,0].boxplot([final_df[final_df['architecture'] == arch]['final_reward'].values 
                          for arch in architectures], labels=architectures)
        axes[1,0].set_title('Final Performance Distribution')
        axes[1,0].set_ylabel('Final Reward')
        axes[1,0].grid(True, alpha=0.3)
    
    # Learning speed comparison
    axes[1,1].text(0.5, 0.5, 'Learning speed analysis\nwould go here',
                  horizontalalignment='center', verticalalignment='center',
                  transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].set_title('Learning Speed')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/comparisons/architecture_comparison.png', dpi=150)
    print('Comparison plot saved to results/comparisons/architecture_comparison.png')
else:
    print('No data found for comparison')
"

echo.
echo ========================================
echo Comparison experiment completed!
echo ========================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo Comparison plot: results/comparisons/architecture_comparison.png
echo.
pause