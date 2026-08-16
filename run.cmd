@echo off
REM ============================================================================
REM  ATRIA EchoTrace - one-click launcher for Windows
REM
REM  Double-click this file. It creates an isolated environment on first run,
REM  installs the review tier, starts the server and opens the workstation in a
REM  browser. Requires only a Python 3.11+ interpreter on PATH (or the `py`
REM  launcher). Uses `uv` when available because it is far faster.
REM
REM  Optional AI tier (local MedGemma 1.5 inference, ~8 GB of gated weights):
REM      run.cmd --ai
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV=.venv"
set "EXTRAS="
set "WANT_AI=0"
for %%A in (%*) do (
    if /I "%%A"=="--ai" set "WANT_AI=1"
)
if "%WANT_AI%"=="1" set "EXTRAS=[ai]"

echo.
echo   ATRIA EchoTrace - starting up
echo   ---------------------------------------------------------------
if "%WANT_AI%"=="1" (
    echo   Tier: review + AI  ^(torch/transformers/peft will be installed^)
) else (
    echo   Tier: review       ^(add --ai for local model inference^)
)
echo.

REM ---- Prefer uv, which manages the interpreter and environment itself -------
where uv >nul 2>&1
if %ERRORLEVEL%==0 (
    echo   Using uv.
    uv venv --allow-existing "%VENV%" || goto :venvfail
    if "%WANT_AI%"=="1" (
        REM PyPI serves CPU-only torch wheels on Windows, so a plain install leaves an
        REM NVIDIA GPU unused. --torch-backend probes the driver and picks the matching
        REM CUDA index, falling back to CPU when there is no GPU. Override with
        REM ATRIA_TORCH_BACKEND=cu118 (older drivers) or =cpu (force CPU).
        if not defined ATRIA_TORCH_BACKEND set "ATRIA_TORCH_BACKEND=auto"
        echo   Selecting a PyTorch build ^(--torch-backend=!ATRIA_TORCH_BACKEND!^) ...
        uv pip install --python "%VENV%\Scripts\python.exe" --torch-backend=!ATRIA_TORCH_BACKEND! -e ".%EXTRAS%" || goto :installfail
    ) else (
        uv pip install --python "%VENV%\Scripts\python.exe" -e ".%EXTRAS%" || goto :installfail
    )
    "%VENV%\Scripts\python.exe" -m atria_echotrace serve
    goto :end
)

REM ---- Fall back to the standard library venv -------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo   ERROR: No Python interpreter found.
    echo   Install Python 3.11 or newer from https://www.python.org/downloads/
    echo   and be sure to tick "Add Python to PATH".
    goto :fail
)

if not exist "%VENV%\Scripts\python.exe" (
    echo   Creating environment in %VENV% ...
    %PY% -m venv "%VENV%" || goto :venvfail
)

set "VPY=%VENV%\Scripts\python.exe"

REM Install only when the entry point is missing, so restarts are instant.
if not exist "%VENV%\Scripts\atria.exe" goto :install
if "%WANT_AI%"=="1" (
    "%VPY%" -c "import torch" >nul 2>&1 || goto :install
)
goto :run

:install
echo   Installing dependencies (first run may take a few minutes) ...
"%VPY%" -m pip install --quiet --upgrade pip
if "%WANT_AI%"=="1" (
    REM Same trap as above, without uv to solve it: PyPI's Windows torch wheel has no
    REM CUDA. Install torch from the PyTorch index first when a driver is present, so
    REM the later editable install finds the requirement already satisfied.
    if not defined ATRIA_TORCH_BACKEND set "ATRIA_TORCH_BACKEND=cu126"
    where nvidia-smi >nul 2>&1
    if !ERRORLEVEL!==0 (
        if /I not "!ATRIA_TORCH_BACKEND!"=="cpu" (
            echo   NVIDIA driver detected; installing CUDA PyTorch ^(!ATRIA_TORCH_BACKEND!^) ...
            "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/!ATRIA_TORCH_BACKEND! || goto :installfail
        )
    ) else (
        echo   No NVIDIA driver detected; installing the CPU PyTorch build.
    )
)
"%VPY%" -m pip install -e ".%EXTRAS%" || goto :installfail

:run
echo.
"%VPY%" -m atria_echotrace serve
goto :end

:venvfail
echo   ERROR: Could not create the virtual environment.
goto :fail

:installfail
echo.
echo   ERROR: Dependency installation failed. See the messages above.
if "%WANT_AI%"=="1" (
    echo   The AI tier downloads PyTorch, which is large; check disk space and network.
)
goto :fail

:fail
echo.
pause
exit /b 1

:end
endlocal
