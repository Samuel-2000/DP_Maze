@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Quick Test with Visualization
echo ========================================

echo.
echo Available models in models/ directory:
dir /b models\*.pt

echo.
set /p MODEL_PATH="Enter model path (or press Enter for latest): "

if "%MODEL_PATH%"=="" (
    echo Searching for latest model...
    for /f "delims=" %%i in ('dir /b /o-d models\*.pt') do (
        set LATEST_MODEL=%%i
        goto found
    )
    :found
    set MODEL_PATH=models\!LATEST_MODEL!
    echo Using latest model: !MODEL_PATH!
)

echo.
echo Testing model with visualization...
python run.py test --model "%MODEL_PATH%" --episodes 3 --visualize

echo.
echo ========================================
echo Test completed!
echo ========================================
pause