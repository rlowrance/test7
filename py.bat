::@echo off
setlocal
set pythonpath   C:\Users\roylo\Dropbox\ads\python_lib
flake8 %1
if %errorlevel% equ 0 python %*
endlocal
