@echo off
echo ========================================
echo Running benchmark on all trained models
echo ========================================

echo.
echo Starting benchmark with 50 episodes per model...
python run.py benchmark --benchmark-episodes 50 --output-dir results/benchmarks

echo.
echo ========================================
echo Benchmark completed!
echo Results saved to: results/benchmarks/
echo ========================================
pause