@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente Python assente. Segui le istruzioni in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app_pioggia.py --server.port 64001 --server.headless false
pause
