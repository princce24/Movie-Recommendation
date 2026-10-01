@echo off
title HCL Movie Recommendation System - Web Application
color 0A
cd /d "%~dp0"

echo ===============================================================================
echo                HCL MOVIE RECOMMENDATION SYSTEM
echo           Item-Item Collaborative Filtering (Streamlit UI)
echo ===============================================================================
echo.

:: 1. Check Python installation
echo [*] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python is not installed or not added to your system PATH!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
python --version

:: 2. Check if model files exist
if not exist "model\movies.pkl" (
    color 0E
    echo.
    echo [!] Pre-trained model artifacts not found in model\ folder!
    echo [*] Running train_model.py to generate artifacts...
    echo.
    python train_model.py
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Model training failed! Please check dataset files in data\ folder.
        pause
        exit /b 1
    )
)

:: 3. Check Streamlit installation
echo [*] Checking Streamlit...
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Streamlit not found. Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to install required packages!
        pause
        exit /b 1
    )
)

:: 4. Start the Streamlit application
echo.
echo ===============================================================================
echo [*] Launching Streamlit Web Application...
echo [*] URL: http://localhost:8501
echo [*] Press Ctrl+C in this window to stop the server when done.
echo ===============================================================================
echo.

streamlit run app.py --server.address 0.0.0.0 --server.port 8501
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Streamlit encountered an issue while running.
    pause
)
