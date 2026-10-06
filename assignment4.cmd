@echo off
setlocal
if defined ASSIGNMENT_PYTHON goto custom
if exist "%~dp0.venv\Scripts\python.exe" goto venv
if exist "%LOCALAPPDATA%\Python\bin\python.exe" goto native
set "TASK_ASSIGNMENT_PYTHON=python"
goto run
:custom
set "TASK_ASSIGNMENT_PYTHON=%ASSIGNMENT_PYTHON%"
goto run
:native
set "TASK_ASSIGNMENT_PYTHON=%LOCALAPPDATA%\Python\bin\python.exe"
goto run
:venv
set "TASK_ASSIGNMENT_PYTHON=%~dp0.venv\Scripts\python.exe"
:run
"%TASK_ASSIGNMENT_PYTHON%" "%~dp0train_assignment4.py" %*
exit /b %errorlevel%

