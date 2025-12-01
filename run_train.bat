@echo off
echo Training memory maze agent...
cd /d "%~dp0"
python run.py --mode train --train-epochs 1000
pause