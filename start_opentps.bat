@echo off
REM Launcher for OpenTPS GUI using pyMatRad conda environment
REM Usage: start_opentps.bat [example1|example2]
set PYMATRAD_PYTHON=C:\Users\jkim20\AppData\Local\anaconda3\envs\pyMatRad\python.exe
set SCRIPT=%~dp0start_opentps.py
%PYMATRAD_PYTHON% "%SCRIPT%" %1 %2 %3
