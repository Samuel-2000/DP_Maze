@echo off
echo ========================================
echo Memory Maze RL - Complete Pipeline
echo ========================================

REM Create necessary directories
echo.
echo Creating directories...
if not exist models mkdir models
if not exist logs mkdir logs
if not exist results\benchmarks mkdir results\benchmarks
if not exist results\plots mkdir results\plots
if not exist results\videos mkdir results\videos

echo.
echo ========================================
echo Step 1: Training models
echo ========================================

REM Train LSTM
echo.
echo Training LSTM model...
python run.py train --network-type lstm --epochs 3000 --batch-size 32 --experiment-name "lstm_quick"

REM Train Transformer
echo.
echo Training Transformer model...
python run.py train --network-type transformer --epochs 3000 --batch-size 16 --auxiliary-tasks --experiment-name "transformer_quick"

echo.
echo ========================================
echo Step 2: Running benchmarks
echo ========================================

echo.
echo Benchmarking trained models...
python run.py benchmark --benchmark-episodes 20 --output-dir results/benchmarks

echo.
echo ========================================
echo Step 3: Creating visualizations
echo ========================================

echo.
echo Creating visualizations...
for %%f in (models\*.pt) do (
    echo Testing: %%~nxf
    python run.py visualize --model "%%f" --episodes 2 --save-gif
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Output directories:
echo - Models:        models/
echo - Logs:          logs/
echo - Benchmarks:    results/benchmarks/
echo - Plots:         results/plots/
echo - Videos:        results/videos/
echo.
pause