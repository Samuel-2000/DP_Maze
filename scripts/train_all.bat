@echo off
echo ========================================
echo Training all memory architectures
echo ========================================

echo.
echo [1/3] Training LSTM baseline...
python run.py train --network-type lstm --epochs 5000 --experiment-name "lstm_baseline"

echo.
echo [2/3] Training Transformer with auxiliary tasks...
python run.py train --network-type transformer --epochs 5000 --auxiliary-tasks --experiment-name "transformer_aux"

echo.
echo [3/3] Training MultiMemory with auxiliary tasks...
python run.py train --network-type multimemory --epochs 5000 --auxiliary-tasks --experiment-name "multimemory_aux"

echo.
echo ========================================
echo All models trained successfully!
echo ========================================
pause