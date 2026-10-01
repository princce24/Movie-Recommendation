@echo off
title HCL Movie Recommendation System - Model Training Pipeline
color 0B
cd /d "%~dp0"

echo ===============================================================================
echo                HCL MOVIE RECOMMENDATION SYSTEM
echo           Collaborative Filtering Model Training Pipeline
echo ===============================================================================
echo.

:: 1. Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not added to your system PATH!
    pause
    exit /b 1
)

echo [*] Starting training pipeline on Netflix dataset...
echo [*] This will build the sparse CSR matrix and calculate item-item cosine similarity.
echo.

python train_model.py

if %errorlevel% equ 0 (
    color 0A
    echo.
    echo ===============================================================================
    echo [SUCCESS] Training complete! All model artifacts generated in model\ folder.
    echo You can now launch the web app by running run_app.bat.
    echo ===============================================================================
) else (
    color 0C
    echo.
    echo ===============================================================================
    echo [ERROR] Training encountered an error. Please inspect the output above.
    echo ===============================================================================
)

echo.
pause
