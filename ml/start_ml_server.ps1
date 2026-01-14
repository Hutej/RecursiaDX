# RecursiaDx ML API Startup Script for Windows (PyTorch)
# This script sets up and starts the ML API server with PyTorch support

param(
    [string]$Host = "localhost",
    [int]$Port = 5000,
    [switch]$Debug,
    [int]$Workers = 1,
    [string]$LogLevel = "INFO",
    [switch]$SkipChecks,
    [switch]$InstallDeps,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host "RecursiaDx ML API Startup Script" -ForegroundColor Green
    Write-Host "================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Parameters:"
    Write-Host "  -Host <string>     Host to bind to (default: localhost)"
    Write-Host "  -Port <int>        Port to bind to (default: 5000)"
    Write-Host "  -Debug             Enable debug mode"
    Write-Host "  -Workers <int>     Number of worker processes (default: 1)"
    Write-Host "  -LogLevel <string> Logging level (default: INFO)"
    Write-Host "  -SkipChecks        Skip dependency and model checks"
    Write-Host "  -InstallDeps       Install Python dependencies"
    Write-Host "  -Help              Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start_ml_server.ps1                    # Start with defaults"
    Write-Host "  .\start_ml_server.ps1 -Debug             # Start in debug mode"
    Write-Host "  .\start_ml_server.ps1 -InstallDeps       # Install dependencies first"
    Write-Host "  .\start_ml_server.ps1 -Port 8080         # Use different port"
    exit 0
}

Write-Host "🚀 RecursiaDx ML API Startup" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

# Check if Python is available
Write-Host "🐍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Change to ML directory
$mlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $mlDir

# Install dependencies if requested
if ($InstallDeps) {
    Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow
    try {
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    }
}

# Check if virtual environment should be used
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔍 Virtual environment found, activating..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} elseif (Test-Path "..\venv\Scripts\Activate.ps1") {
    Write-Host "🔍 Virtual environment found in parent directory, activating..." -ForegroundColor Yellow
    & "..\venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
}

# Build Python command arguments
$pythonArgs = @("start_server.py")
$pythonArgs += "--host", $Host
$pythonArgs += "--port", $Port
$pythonArgs += "--log-level", $LogLevel

if ($Debug) {
    $pythonArgs += "--debug"
}

if ($Workers -gt 1) {
    $pythonArgs += "--workers", $Workers
}

if ($SkipChecks) {
    $pythonArgs += "--skip-checks"
}

# Display startup information
Write-Host ""
Write-Host "🌐 Server Configuration:" -ForegroundColor Cyan
Write-Host "   Host: $Host" -ForegroundColor White
Write-Host "   Port: $Port" -ForegroundColor White
Write-Host "   Debug: $Debug" -ForegroundColor White
Write-Host "   Workers: $Workers" -ForegroundColor White
Write-Host "   Log Level: $LogLevel" -ForegroundColor White
Write-Host ""

# Start the server
Write-Host "🚀 Starting ML API server..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

try {
    & python $pythonArgs
} catch {
    Write-Host "❌ Failed to start server" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "👋 Server stopped" -ForegroundColor Yellow