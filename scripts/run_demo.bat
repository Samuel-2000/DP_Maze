@echo off
echo ========================================
echo Memory Maze RL Demo
echo ========================================

echo.
echo Starting interactive demo...

:menu
cls
echo ========================================
echo MEMORY MAZE RL - MAIN MENU
echo ========================================
echo.
echo 1. Quick training demo (500 epochs)
echo 2. Test existing model
echo 3. Benchmark all models
echo 4. Compare architectures
echo 5. Watch agent play
echo 6. Exit
echo.
set /p CHOICE="Enter choice (1-6): "

if "%CHOICE%"=="1" goto train_demo
if "%CHOICE%"=="2" goto test_demo
if "%CHOICE%"=="3" goto benchmark_demo
if "%CHOICE%"=="4" goto compare_demo
if "%CHOICE%"=="5" goto watch_demo
if "%CHOICE%"=="6" goto exit_demo

echo Invalid choice, please try again.
pause
goto menu

:train_demo
echo.
echo Starting quick training demo...
python run.py train --network-type lstm --epochs 500 --batch-size 16 --experiment-name "demo"
pause
goto menu

:test_demo
echo.
dir /b models\*.pt
echo.
set /p MODEL="Enter model filename: "
python run.py test --model "models\%MODEL%" --episodes 3 --visualize
pause
goto menu

:benchmark_demo
echo.
echo Running benchmark...
python run.py benchmark --benchmark-episodes 10
pause
goto menu

:compare_demo
echo.
echo Comparing architectures...
python run.py compare --architectures lstm transformer --epochs 1000 --trials 2
pause
goto menu

:watch_demo
echo.
set /p MODEL="Enter model filename (or press Enter for latest): "
if "%MODEL%"=="" (
    for /f "delims=" %%i in ('dir /b /o-d models\*.pt') do set LATEST_MODEL=%%i
    set MODEL=!LATEST_MODEL!
)
python run.py visualize --model "models\%MODEL%" --episodes 2 --save-gif
pause
goto menu

:exit_demo
echo.
echo Thank you for using Memory Maze RL!
echo ========================================
pause