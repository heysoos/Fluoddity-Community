@echo off
rem Stop Fluoddity. Asks it to quit first, force-kills only if that fails.
rem
rem The graceful step matters: a force-kill skips releaseSender(), leaving a
rem stale entry in Spout's registry. The next run then finds its name taken and
rem is silently renamed to "fluoddity_1" -- which a receiver watching
rem "fluoddity" never sees. Black texture, no error anywhere.
rem
rem The force step is deliberately NOT "taskkill /F /IM python.exe": that would
rem also kill the gfx\Neuro TensorFlow processes and any other Python you have.

set FLUO_PY=python
set OSC_PORT=5011

rem 1. Ask nicely over the OSC channel the app already listens on.
%FLUO_PY% -c "from pythonosc import udp_client; udp_client.SimpleUDPClient('127.0.0.1', %OSC_PORT%).send_message('/fluoddity/cmd', 'quit')" 2>nul

rem 2. Give it a moment to release the Spout sender and exit.
ping -n 3 127.0.0.1 >nul

rem 3. Anything left is stuck -- force it, matching on the command line only.
powershell -NoProfile -Command ^
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*--spout-out fluoddity*' } | ForEach-Object { Write-Host 'force-killing stuck process' $_.ProcessId; Stop-Process -Id $_.ProcessId -Force }"
