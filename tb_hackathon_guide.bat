@echo off
echo ======================================================
echo TB Detection with Federated Learning - Guide
echo Techkriti 2025 ML Hackathon - IIT Kanpur
echo ======================================================
echo.

echo This guide will help you set up and run the TB detection system.
echo.

:menu
echo Choose an option:
echo 1. Set up project directories
echo 2. Convert and prepare dataset
echo 3. Train the model with Federated Learning
echo 4. Make predictions on test data
echo 5. Launch interactive web interface
echo 6. Exit
echo.

set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto convert
if "%choice%"=="3" goto train
if "%choice%"=="4" goto predict
if "%choice%"=="5" goto streamlit
if "%choice%"=="6" goto end

echo Invalid choice. Please try again.
goto menu

:setup
echo.
echo Setting up project directories...
python main.py setup
echo.
echo Project directories have been set up successfully!
echo.
pause
goto menu

:convert
echo.
echo === Dataset Conversion ===
echo.
echo This will convert your TB X-ray dataset into the required format.
echo You need to specify the source directory containing your dataset.
echo.
set /p source_dir=Enter the path to your source dataset directory: 
echo.
echo Conversion Options:
echo 1. Auto-detect dataset format (recommended)
echo 2. Folder-based dataset (with TB/Normal subfolders)
echo 3. CSV-based dataset (with image paths and labels)
echo.
set /p format_choice=Enter your choice (1-3): 

if "%format_choice%"=="1" (
    set format=auto
) else if "%format_choice%"=="2" (
    set format=folders
) else if "%format_choice%"=="3" (
    set format=csv
) else (
    echo Invalid choice. Using auto-detect.
    set format=auto
)

echo.
echo Converting dataset...
python convert_dataset.py --source_dir "%source_dir%" --target_dir data --format %format%
echo.
echo Dataset conversion completed! Images have been placed in data/train/normal, data/train/tb, and data/test directories.
echo.
pause
goto menu

:train
echo.
echo === Training the Model ===
echo.
echo Choose training method:
echo 1. Train with Federated Learning (simulates training across 3 hospitals)
echo 2. Train without Federated Learning (centralized training)
echo.
set /p train_choice=Enter your choice (1-2): 

if "%train_choice%"=="1" (
    set federated=true
) else if "%train_choice%"=="2" (
    set federated=false
) else (
    echo Invalid choice. Using Federated Learning.
    set federated=true
)

echo.
set /p rounds=Enter number of federated rounds (default: 3): 
if "%rounds%"=="" set rounds=3

set /p epochs=Enter number of epochs per round (default: 10): 
if "%epochs%"=="" set epochs=10

echo.
echo Starting training...
python main.py train --federated %federated% --rounds %rounds% --epochs %epochs%
echo.
echo Training completed!
echo.
pause
goto menu

:predict
echo.
echo === Generating Predictions ===
echo.
echo This will generate predictions for all images in the data/test directory.
echo.
set /p visualize=Number of test samples to visualize with GradCAM (default: 5): 
if "%visualize%"=="" set visualize=5

echo.
echo Generating predictions...
python main.py predict --visualize_samples %visualize%
echo.
echo Predictions completed!
echo A submission.csv file has been created with the results.
echo GradCAM visualizations have been saved in the visualizations directory.
echo.
pause
goto menu

:streamlit
echo.
echo === Launching Web Interface ===
echo.
echo This will start a Streamlit web application where you can:
echo - Upload chest X-rays for TB detection
echo - View model's prediction results
echo - See GradCAM visualizations
echo.
echo Press Ctrl+C in the terminal to stop the application when done.
echo.
pause
echo Launching web interface...
python main.py streamlit
goto menu

:end
echo.
echo Thank you for using the TB Detection system!
echo.
pause
exit 