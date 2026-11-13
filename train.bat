@echo off
echo ========================================================
echo   Starting DeepGuard AI - Model Training Pipeline
echo   Dataset: 4,000 images (2,800 train, 600 val, 600 test)
echo   Model: Hybrid CNN-Transformer (EfficientNet-B4 + ViT)
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python scripts/train_model.py --epochs 10 --batch_size 16 --model_type hybrid
pause

