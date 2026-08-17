# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Fluoddity particle simulation

Usage:
    pyinstaller Fluoddity.spec

This will create a dist/Fluoddity folder with all dependencies bundled.
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = os.path.abspath(SPECPATH)

# Find GLFW DLLs
def collect_glfw_binaries():
    """Collect GLFW native libraries."""
    binaries = []
    try:
        import glfw
        glfw_path = Path(glfw.__file__).parent
        # Look for DLL files in the glfw package directory
        for dll_file in glfw_path.rglob('*.dll'):
            binaries.append((str(dll_file), '.'))
        print(f"Found {len(binaries)} GLFW binaries")
    except Exception as e:
        print(f"Warning: Could not collect GLFW binaries: {e}")
    return binaries

glfw_binaries = collect_glfw_binaries()

# Find ffmpeg for video recording
def collect_ffmpeg():
    """Collect ffmpeg executable if available."""
    # Check if we should skip bundling ffmpeg (set by build.ps1 -NoFfmpeg)
    if os.environ.get('FLUODDITY_NO_FFMPEG'):
        print("Skipping ffmpeg bundling (-NoFfmpeg flag set)")
        print("         Users will need ffmpeg in their PATH for video recording")
        return []

    binaries = []
    try:
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            binaries.append((ffmpeg_path, '.'))
            print(f"Found ffmpeg at: {ffmpeg_path}")
        else:
            print("Warning: ffmpeg not found in PATH - video recording will not work in the built application")
            print("         To enable video recording, install ffmpeg and rebuild, or users can install it separately")
    except Exception as e:
        print(f"Warning: Could not check for ffmpeg: {e}")
    return binaries

ffmpeg_binaries = collect_ffmpeg()

# Collect all shader files
shader_files = []
shader_dir = Path(project_root) / 'shaders'
if shader_dir.exists():
    for shader_file in shader_dir.glob('*'):
        if shader_file.is_file():
            shader_files.append((str(shader_file), 'shaders'))

# Additional data files
datas = shader_files + [
    # Add any other data files here if needed
]

# Hidden imports (packages that PyInstaller might not detect automatically)
hiddenimports = [
    'moderngl',
    'glfw',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL.ImageGrab',
    'PIL.ImageDraw',
    'PIL.ImageFont',
    'imgui_bundle',
    'imgui_bundle.imgui',
    'imgui_bundle.python_backends',
    'imgui_bundle.python_backends.glfw_backend',
    'pyaudiowpatch',
]

a = Analysis(
    ['launcher_debug.py'],  # Use debug launcher to catch startup errors
    pathex=[project_root],
    binaries=glfw_binaries + ffmpeg_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cv2', 'opencv'],  # imgui_bundle has optional cv2 support we don't use
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Fluoddity',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False for windowed mode (no console)
    disable_windowing_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Fluoddity',
)
