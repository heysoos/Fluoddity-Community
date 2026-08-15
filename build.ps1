# Build script for Fluoddity
# This script activates the virtual environment (if needed), runs PyInstaller, and fixes shader paths

param(
    [switch]$NoFfmpeg
)

Write-Host "=== Fluoddity Build Script ===" -ForegroundColor Cyan
Write-Host ""

# Set environment variable for PyInstaller spec file
if ($NoFfmpeg) {
    Write-Host "Building WITHOUT bundled ffmpeg (users must have ffmpeg in PATH)" -ForegroundColor Yellow
    $env:FLUODDITY_NO_FFMPEG = "1"
} else {
    $env:FLUODDITY_NO_FFMPEG = ""
}

# Step 1: Check if virtual environment is already activated, if not activate it
if ($env:VIRTUAL_ENV) {
    Write-Host "[1/5] Virtual environment already activated: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Yellow
    & ".\Scratch.venv\Scripts\Activate.ps1"
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "Error: Failed to activate virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Run PyInstaller
Write-Host "[2/5] Running PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller --clean --noconfirm Fluoddity.spec
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: PyInstaller build failed" -ForegroundColor Red
    exit 1
}

# Step 3: Move shaders folder to correct location
Write-Host "[3/5] Moving shaders folder..." -ForegroundColor Yellow
$shadersSource = "dist\Fluoddity\_internal\shaders"
$shadersDestination = "dist\Fluoddity\shaders"

if (Test-Path $shadersSource) {
    # Remove destination if it exists
    if (Test-Path $shadersDestination) {
        Remove-Item -Recurse -Force $shadersDestination
    }
    # Move shaders folder
    Move-Item -Path $shadersSource -Destination $shadersDestination
    Write-Host "Shaders folder moved successfully" -ForegroundColor Green
} else {
    Write-Host "Warning: Shaders folder not found at $shadersSource" -ForegroundColor Yellow
}

# Step 4: Copy default configuration files
Write-Host "[4/5] Copying default configuration files..." -ForegroundColor Yellow

# Copy default_keyboard_controls.json
$keyboardSource = "default_keyboard_controls.json"
$keyboardDest = "dist\Fluoddity\default_keyboard_controls.json"
if (Test-Path $keyboardSource) {
    Copy-Item -Path $keyboardSource -Destination $keyboardDest -Force
    Write-Host "  Copied default_keyboard_controls.json" -ForegroundColor Green
} else {
    Write-Host "  Warning: default_keyboard_controls.json not found" -ForegroundColor Yellow
}

# Copy default_imgui.ini
$imguiSource = "default_imgui.ini"
$imguiDest = "dist\Fluoddity\default_imgui.ini"
if (Test-Path $imguiSource) {
    Copy-Item -Path $imguiSource -Destination $imguiDest -Force
    Write-Host "  Copied default_imgui.ini" -ForegroundColor Green
} else {
    Write-Host "  Warning: default_imgui.ini not found" -ForegroundColor Yellow
}

# Step 5: Copy physics_configs (Core and Advanced folders only - user configs stay in Documents)
Write-Host "[5/5] Copying bundled physics_configs..." -ForegroundColor Yellow

$physicsSource = "physics_configs"
$physicsDest = "dist\Fluoddity\physics_configs"

# Create destination directory if it doesn't exist
if (-not (Test-Path $physicsDest)) {
    New-Item -ItemType Directory -Path $physicsDest -Force | Out-Null
}

# Copy Core folder (bundled presets)
$coreSource = "$physicsSource\Core"
$coreDest = "$physicsDest\Core"
if (Test-Path $coreSource) {
    if (Test-Path $coreDest) {
        Remove-Item -Recurse -Force $coreDest
    }
    Copy-Item -Path $coreSource -Destination $coreDest -Recurse -Force
    $coreCount = (Get-ChildItem -Path $coreDest -Filter "*.json" -File).Count
    Write-Host "  Copied physics_configs/Core ($coreCount configs)" -ForegroundColor Green
} else {
    Write-Host "  Warning: physics_configs/Core not found" -ForegroundColor Yellow
}

# Copy Advanced folder (bundled presets)
$advancedSource = "$physicsSource\Advanced"
$advancedDest = "$physicsDest\Advanced"
if (Test-Path $advancedSource) {
    if (Test-Path $advancedDest) {
        Remove-Item -Recurse -Force $advancedDest
    }
    Copy-Item -Path $advancedSource -Destination $advancedDest -Recurse -Force
    $advancedCount = (Get-ChildItem -Path $advancedDest -Filter "*.json" -File).Count
    Write-Host "  Copied physics_configs/Advanced ($advancedCount configs)" -ForegroundColor Green
} else {
    Write-Host "  Warning: physics_configs/Advanced not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Build Complete! ===" -ForegroundColor Green
Write-Host "Executable location: dist\Fluoddity\Fluoddity.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "User data will be stored in: Documents\Fluoddity\" -ForegroundColor Cyan
Write-Host "  - preferences.config" -ForegroundColor Gray
Write-Host "  - keyboard_controls.json" -ForegroundColor Gray
Write-Host "  - physics_configs\ (user-created)" -ForegroundColor Gray
Write-Host "  - Screenshots\" -ForegroundColor Gray
Write-Host "  - Videos\" -ForegroundColor Gray
Write-Host "  - imgui.ini (window layout, seeded from default_imgui.ini)" -ForegroundColor Gray
Write-Host ""
Write-Host "To test the build, run: .\dist\Fluoddity\Fluoddity.exe" -ForegroundColor Cyan
