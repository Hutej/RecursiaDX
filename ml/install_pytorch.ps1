# PowerShell script to install PyTorch and dependencies for RecursiaDx
# Run this script to set up the ML environment with PyTorch

Write-Host "🚀 Installing PyTorch and dependencies for RecursiaDx..." -ForegroundColor Green
Write-Host "=" * 60

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ Pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Pip not found. Please install pip first." -ForegroundColor Red
    exit 1
}

# Detect CUDA availability (optional)
Write-Host "`n🔍 Checking for CUDA..." -ForegroundColor Yellow
try {
    $cudaVersion = nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>$null
    if ($cudaVersion) {
        Write-Host "✅ NVIDIA GPU detected. Installing PyTorch with CUDA support..." -ForegroundColor Green
        $torchIndex = "--index-url https://download.pytorch.org/whl/cu121"
    } else {
        Write-Host "⚠️ No NVIDIA GPU detected. Installing CPU-only PyTorch..." -ForegroundColor Yellow
        $torchIndex = "--index-url https://download.pytorch.org/whl/cpu"
    }
} catch {
    Write-Host "⚠️ Could not detect GPU. Installing CPU-only PyTorch..." -ForegroundColor Yellow
    $torchIndex = "--index-url https://download.pytorch.org/whl/cpu"
}

# Install PyTorch first
Write-Host "`n📦 Installing PyTorch..." -ForegroundColor Cyan
try {
    if ($torchIndex) {
        pip install torch torchvision torchaudio $torchIndex
    } else {
        pip install torch torchvision torchaudio
    }
    Write-Host "✅ PyTorch installation completed" -ForegroundColor Green
} catch {
    Write-Host "❌ PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Install other requirements
Write-Host "`n📦 Installing other dependencies..." -ForegroundColor Cyan
try {
    pip install -r requirements.txt
    Write-Host "✅ All dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Dependencies installation failed" -ForegroundColor Red
    exit 1
}

# Test installation
Write-Host "`n🧪 Testing PyTorch installation..." -ForegroundColor Cyan
try {
    python test_pytorch.py
    Write-Host "✅ PyTorch test completed" -ForegroundColor Green
} catch {
    Write-Host "❌ PyTorch test failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Setup completed successfully!" -ForegroundColor Green
Write-Host "You can now run the ML server with:" -ForegroundColor Yellow
Write-Host "  python start_server.py" -ForegroundColor White
Write-Host "`nOr test the installation with:" -ForegroundColor Yellow
Write-Host "  python test_pytorch.py" -ForegroundColor White