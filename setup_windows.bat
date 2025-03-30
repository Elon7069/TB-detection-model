@echo off
echo ======================================================
echo TB Detection - Windows Setup Helper
echo ======================================================
echo.

echo This script will help you set up the TB Detection project on Windows.
echo.

:menu
echo Choose an option:
echo 1. Run in demo mode (recommended)
echo 2. Try to install TensorFlow for Windows
echo 3. Exit
echo.

set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" goto demo_mode
if "%choice%"=="2" goto install_tensorflow
if "%choice%"=="3" goto end

echo Invalid choice. Please try again.
goto menu

:demo_mode
echo.
echo Setting up demo mode...
echo.

echo Installing required dependencies...
pip install numpy pandas scikit-learn matplotlib
if %ERRORLEVEL% neq 0 (
    echo Failed to install basic dependencies.
    echo Please make sure you have Python and pip installed correctly.
    goto end
)

echo.
echo Setting up the demo environment...
python run_demo.py --mode setup

echo.
echo Demo mode setup completed!
echo.
echo To run the TB Detection system in demo mode:
echo   1. Train the model:      python run_demo.py --mode train
echo   2. Generate predictions: python run_demo.py --mode predict
echo   3. Launch web interface: python run_demo.py --mode streamlit
echo.
echo For more information, see DEMO_MODE.md
echo.
pause
goto end

:install_tensorflow
echo.
echo Attempting to install TensorFlow for Windows...
echo.

echo First, checking your Python version...
python --version
if %ERRORLEVEL% neq 0 (
    echo Python not found.
    echo Please install Python 3.8-3.10 for best compatibility with TensorFlow.
    goto end
)

echo.
echo Your Python version might not be compatible with TensorFlow.
echo TensorFlow typically works best with Python 3.8-3.10.
echo.
echo Do you want to try installing TensorFlow anyway? (y/n)
set /p tf_choice=

if not "%tf_choice%"=="y" (
    echo.
    echo Skipping TensorFlow installation.
    echo We recommend using the demo mode instead.
    goto setup_demo_mode
)

echo.
echo Installing tensorflow-cpu (version 2.10.0 for best Windows compatibility)...
pip install tensorflow==2.10.0
if %ERRORLEVEL% neq 0 (
    echo Failed to install TensorFlow 2.10.0.
    echo.
    echo You might need a different Python version (3.8-3.10 is recommended).
    echo.
    echo Trying to install tensorflow-directml-plugin for Windows...
    pip install tensorflow==2.10.0 tensorflow-directml-plugin==0.4.0
    if %ERRORLEVEL% neq 0 (
        echo.
        echo Failed to install TensorFlow with DirectML.
        echo We recommend using the demo mode instead.
        goto setup_demo_mode
    )
)

echo.
echo Installing other dependencies...
pip install numpy pandas scikit-learn matplotlib streamlit pillow
if %ERRORLEVEL% neq 0 (
    echo Warning: Some dependencies failed to install.
    echo You may still be able to run the system, but with limited functionality.
)

echo.
echo Setting up project directories...
python main.py setup

echo.
echo Setup completed!
echo.
echo You can now run:
echo   python main.py train
echo   python main.py predict
echo   python main.py streamlit
echo.
echo If you encounter any issues, try using the demo mode instead:
echo   python run_demo.py --mode setup
echo.
pause
goto end

:setup_demo_mode
echo.
echo Setting up demo mode instead...
echo.
python run_demo.py --mode setup
echo.
echo Demo mode setup completed!
echo.
echo To run the TB Detection system in demo mode:
echo   1. Train the model:      python run_demo.py --mode train
echo   2. Generate predictions: python run_demo.py --mode predict
echo   3. Launch web interface: python run_demo.py --mode streamlit
echo.
pause
goto end

:end
echo.
echo Thank you for using the TB Detection system!
echo. 