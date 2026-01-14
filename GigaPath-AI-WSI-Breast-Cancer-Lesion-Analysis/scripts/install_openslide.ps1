$ErrorActionPreference = "Stop"

$url = "https://github.com/openslide/openslide-bin/releases/download/v4.0.0.11/openslide-bin-4.0.0.11-windows-x64.zip"
$zipPath = "$PSScriptRoot\openslide.zip"

$localPath = "$PSScriptRoot\..\openslide"

Write-Host "Downloading OpenSlide from $url..."
# GitHub releases might need TLS 1.2+ forced
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile $zipPath

Write-Host "Download complete."

function Install-OpenSlide {
    param (
        [string]$DestinationPath
    )
    
    try {
        Write-Host "Attempting installation to $DestinationPath..."
        if (-not (Test-Path $DestinationPath)) {
            New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
        }
        
        Expand-Archive -Path $zipPath -DestinationPath $DestinationPath -Force
        
        # Find the extracted folder
        $extractedFolder = Get-ChildItem -Path $DestinationPath -Directory | Where-Object { $_.Name -like "openslide-bin-*-windows-x64" } | Select-Object -First 1
        
        if ($extractedFolder) {
            # Normalize to simpler path if possible, but since we might be in C:\ or local, 
            # let's just use the path we have and add bin to PATH.
            $binPath = Join-Path $extractedFolder.FullName "bin"
            return $binPath
        }
        else {
            throw "Extraction failed, folder not found in $DestinationPath"
        }
    }
    catch {
        Write-Warning "Failed to install to $DestinationPath. Error: $_"
        return $null
    }
}

# Try C:\OpenSlide (requires Admin usually for C:\ root writing/creating folders)
# Actually, extracting to C:\ creates C:\openslide-win64-4.0.0.11
# We want C:\OpenSlide to be the folder containing bin? Or just C:\OpenSlide\bin?
# Standard is often C:\openslide-win64-xxx\bin added to path.
# Let's try to extract to C:\ first.

$binPath = $null

try {
    Write-Host "Attempting to extract to C:\..."
    Expand-Archive -Path $zipPath -DestinationPath "C:\" -Force -ErrorAction Stop
    $extractedFolder = Get-ChildItem -Path "C:\" -Directory | Where-Object { $_.Name -like "openslide-bin-*-windows-x64" } | Select-Object -First 1
    
    if ($extractedFolder) {
        $binPath = Join-Path $extractedFolder.FullName "bin"
        Write-Host "Successfully extracted to C:\"
    }
    else {
        throw "Could not find extracted folder in C:\"
    }
}
catch {
    Write-Warning "Could not extract to C:\ (likely permission denied). Falling back to local project directory."
    $binPath = Install-OpenSlide -DestinationPath $localPath
}

if ($binPath -and (Test-Path $binPath)) {
    Write-Host "OpenSlide bin found at: $binPath"
    
    # Add to User PATH
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($currentPath -notlike "*$binPath*") {
        Write-Host "Adding to User PATH..."
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$binPath", "User")
        Write-Host "Added to PATH."
    }
    else {
        Write-Host "Already in PATH."
    }

    # Also set strictly for current session so immediate verification works (if possible)
    $env:Path += ";$binPath"
    
    Write-Host "`nInstallation Successful!"
    Write-Host "IMPORTANT: Please restart your terminal/VS Code for the PATH changes to take effect."
    
    # Clean up zip
    Remove-Item $zipPath -Force
}
else {
    Write-Error "Could not install OpenSlide binaries."
}
