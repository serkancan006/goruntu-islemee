@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Once KURULUM.bat dosyasini calistirin.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m mumvision.gui
if errorlevel 1 pause
