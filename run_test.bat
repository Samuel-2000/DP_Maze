@echo off
echo Testing memory maze agent...
cd /d "%~dp0"
python run.py --mode test --model models/policy_epoch_013000.pt --visualize --test-episodes 20 --max-age 1000
pause