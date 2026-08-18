@echo off
rem Launch Fluoddity as a vvvv source. Called from ShellExecute (Windows),
rem mirroring gfx\Neuro\PGAN\SpoutPGAN.bat.
rem
rem Usage:  Fluoddity.bat [width] [height] [extra args...]
rem   Fluoddity.bat 1920 1080
rem   Fluoddity.bat 1280 720 --offscreen
rem
rem The cd below is not optional. Fluoddity loads its shaders by RELATIVE path
rem (read_shader('shaders/camera.vert'), camera.py:36), so it must run with the
rem repo root as the working directory or every shader fails to compile.

set WIDTH=%1
set HEIGHT=%2
if "%WIDTH%"=="" set WIDTH=1280
if "%HEIGHT%"=="" set HEIGHT=720

set FLUO_DIR=%3
if "%FLUO_DIR%"=="" set FLUO_DIR=C:\_work\dev\fluo\Fluoddity-Community

rem Drop the three positional args so %4.. can carry extra flags through.
shift
shift
shift

pushd "%FLUO_DIR%"

python main.py ^
--spout-out fluoddity ^
--spout-in vvvv_fluo ^
--osc-port 5011 ^
--osc-return-port 5012 ^
--width %WIDTH% --height %HEIGHT% ^
%1 %2 %3 %4 %5 %6 %7

popd
