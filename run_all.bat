@echo off
REM run_all.bat
echo ========================================
echo Maze RL - Complete Setup and Test
echo ========================================

echo.
echo 1. Creating directory structure...
if not exist src\core mkdir src\core
if not exist src\networks mkdir src\networks
if not exist src\training mkdir src\training
if not exist src\evaluation mkdir src\evaluation

echo.
echo 2. Testing imports...
python run_simple.py --test

echo.
echo 3. Running demo...
python run_simple.py --demo

echo.
echo ========================================
echo Setup completed!
echo ========================================
echo.
echo Next steps:
echo   1. python run_simple.py --test (test all components)
echo   2. python run_simple.py --demo (run quick demo)
echo   3. python run.py train --network lstm (start training)
echo.
pause