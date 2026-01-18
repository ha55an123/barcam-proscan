@echo off
REM Build Barcam-ProScan.exe with PyInstaller

REM Clean previous build/dist
rmdir /s /q build
rmdir /s /q dist

REM PyInstaller command
pyinstaller --onefile --windowed --name "Barcam-ProScan" --icon=app.ico ^
--add-data "README.md;." ^
--add-data "Barcam-ProScan.spec;." ^
--add-data "Logos;." ^
--hidden-import pyzbar ^
--hidden-import cv2 ^
--hidden-import pandas ^
Barcam-ProScan.py

pause
