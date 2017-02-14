@echo off
setlocal
flake8 %1
if %errorlevel% equ 0 python %*
python %*
endlocal
