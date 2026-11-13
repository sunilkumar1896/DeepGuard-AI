@echo off
echo ========================================================
echo   Downloading OpenRL/DeepFakeFace Benchmark Dataset
echo ========================================================
cd /d "%~dp0"

if not exist "data\downloads" mkdir "data\downloads"

echo.
echo [1/2] Downloading REAL Faces (wiki.zip - 1.6 GB)...
curl.exe -L -C - --retry 5 --retry-delay 3 "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip" -o "data\downloads\wiki.zip"

echo.
echo [2/2] Downloading FAKE Faces (insight.zip - 1.0 GB)...
curl.exe -L -C - --retry 5 --retry-delay 3 "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/insight.zip" -o "data\downloads\insight.zip"

echo.
echo Extracting and organizing dataset...
call .venv\Scripts\activate.bat
python scripts\download_deepfakeface.py --max_per_class 5000

echo ========================================================
echo   Download and Dataset Integration Complete!
echo ========================================================
pause

