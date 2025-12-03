@echo off
echo ========================================
echo Cleaning up generated files
echo ========================================

echo.
set /p CONFIRM="Are you sure you want to delete all generated files? (y/n): "

if /i "%CONFIRM%"=="y" (
    echo Deleting models...
    del /q models\*.pt 2>nul
    
    echo Deleting logs...
    rmdir /s /q logs 2>nul
    mkdir logs
    
    echo Deleting results...
    rmdir /s /q results 2>nul
    mkdir results
    mkdir results\benchmarks
    mkdir results\plots
    mkdir results\videos
    
    echo Deleting __pycache__ directories...
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    
    echo.
    echo Cleanup completed!
) else (
    echo.
    echo Cleanup cancelled.
)

pause