@echo off
setlocal
set pythonpath   C:\Users\roylo\Dropbox\ads\research\python_lib
flake8 %1
if %errorlevel% equ 0 python %*
python %*
endlocal
