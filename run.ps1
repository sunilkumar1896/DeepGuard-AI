Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting DeepGuard AI - Deepfake Detection Platform" -ForegroundColor Cyan
Write-Host "  Web UI will be available at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
