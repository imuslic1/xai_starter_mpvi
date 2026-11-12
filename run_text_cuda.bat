@echo off
setlocal
cd /d "%~dp0"
title XAI - Text - CUDA

rem Prefer an activated venv; otherwise check project and parent folders.
set "PYTHON="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON if exist "%~dp0..\.venv\Scripts\python.exe" set "PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not defined PYTHON (
  echo [ERROR] No Python virtual environment found.
  echo Activate a configured venv, or create .venv in this folder or its parent.
  goto :error
)

echo.
echo ============================================================
echo  XAI KNOWLEDGE DISTILLATION - TEXT - CUDA
echo ============================================================
echo  Python: "%PYTHON%"
echo  Backend: cuda
echo.

"%PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: install CUDA PyTorch and check NVIDIA drivers'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 (
  echo.
  echo [ERROR] GPU backend check failed. See README setup instructions.
  goto :error
)

echo.
echo [START] Running text.py ...
echo.
rem Extra arguments override argparse defaults (e.g. --skip-train or --epochs 10).
"%PYTHON%" "%~dp0text_reviews.py" --train-size 3000 --epochs 2 --batch 8 %* --device cuda
if errorlevel 1 (
  echo.
  echo [ERROR] Experiment failed. Review the traceback above.
  goto :error
)
echo.
echo [DONE] Experiment completed. Check outputs\text\
goto :finish

:error
set "RESULT=1"
goto :end
:finish
set "RESULT=0"
:end
echo.
pause
exit /b %RESULT%
