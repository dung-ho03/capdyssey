@echo off
cd /d "%~dp0"
py -3 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
if errorlevel 1 goto fail
echo Setup complete. Run run_desktop.bat.
pause
exit /b 0
:fail
echo Setup failed. Install Python 3.11 or newer with Tkinter, then retry.
pause
exit /b 1
