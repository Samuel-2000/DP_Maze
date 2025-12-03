@echo off
echo ========================================
echo Installing dependencies
echo ========================================

echo.
echo Installing core dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo Installing development dependencies...
pip install pytest coverage black isort

echo.
echo ========================================
echo Installation completed!
echo ========================================
pause