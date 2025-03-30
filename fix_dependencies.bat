@echo off
echo ========================================================
echo TB Detection - Fix Dependencies for Windows
echo ========================================================
echo.

echo This script will help fix dependency issues for the TB Detection project on Windows.
echo.

echo Step 1: Checking Python installation...
python --version
if %ERRORLEVEL% neq 0 (
    echo Python not found. Please install Python 3.6 or higher.
    pause
    exit /b 1
)

echo.
echo Step 2: Installing compatible dependencies for Windows...
echo.

echo Installing numpy...
pip install numpy==1.23.5
if %ERRORLEVEL% neq 0 echo Failed to install numpy

echo Installing pandas...
pip install pandas==2.0.1
if %ERRORLEVEL% neq 0 echo Failed to install pandas

echo Installing scikit-learn...
pip install scikit-learn==1.2.2
if %ERRORLEVEL% neq 0 echo Failed to install scikit-learn

echo Installing matplotlib...
pip install matplotlib==3.7.1
if %ERRORLEVEL% neq 0 echo Failed to install matplotlib

echo Installing OpenCV...
pip install opencv-python==4.7.0.72
if %ERRORLEVEL% neq 0 echo Failed to install OpenCV

echo Installing Pillow...
pip install pillow==9.5.0
if %ERRORLEVEL% neq 0 echo Failed to install Pillow

echo Installing Streamlit...
pip install streamlit==1.22.0
if %ERRORLEVEL% neq 0 echo Failed to install Streamlit

echo Installing TensorFlow with DirectML support for Windows...
pip install tensorflow==2.10.0 tensorflow-directml-plugin==0.4.0
if %ERRORLEVEL% neq 0 echo Failed to install TensorFlow

echo Installing other dependencies...
pip install tqdm==4.65.0 seaborn==0.12.2 plotly==5.14.1 flwr==1.4.0 grad-cam==1.4.8
if %ERRORLEVEL% neq 0 echo Failed to install additional dependencies

echo.
echo Step 3: Setting up project directories...
mkdir data\train\normal data\train\tb data\test fl_models visualizations 2>nul
echo Directories created!

echo.
echo ========================================================
echo Dependencies installation completed!
echo.
echo You can now run the TB Detection system:
echo - To train the model: python main.py train
echo - To make predictions: python main.py predict
echo - To launch web interface: python main.py streamlit
echo.
echo Or use the interactive guide: tb_hackathon_guide.bat
echo ========================================================
echo.

pause 