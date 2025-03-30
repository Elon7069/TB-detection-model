#!/bin/bash

echo "======================================================"
echo "TB Detection with Federated Learning - Guide"
echo "Techkriti 2025 ML Hackathon - IIT Kanpur"
echo "======================================================"
echo

echo "This guide will help you set up and run the TB detection system."
echo

function show_menu {
    echo "Choose an option:"
    echo "1. Set up project directories"
    echo "2. Convert and prepare dataset"
    echo "3. Train the model with Federated Learning"
    echo "4. Make predictions on test data"
    echo "5. Launch interactive web interface"
    echo "6. Exit"
    echo
    
    read -p "Enter your choice (1-6): " choice
    
    case $choice in
        1) setup ;;
        2) convert ;;
        3) train ;;
        4) predict ;;
        5) streamlit ;;
        6) exit 0 ;;
        *) echo "Invalid choice. Please try again."; show_menu ;;
    esac
}

function setup {
    echo
    echo "Setting up project directories..."
    python main.py setup
    echo
    echo "Project directories have been set up successfully!"
    echo
    read -p "Press Enter to continue..."
    show_menu
}

function convert {
    echo
    echo "=== Dataset Conversion ==="
    echo
    echo "This will convert your TB X-ray dataset into the required format."
    echo "You need to specify the source directory containing your dataset."
    echo
    read -p "Enter the path to your source dataset directory: " source_dir
    echo
    echo "Conversion Options:"
    echo "1. Auto-detect dataset format (recommended)"
    echo "2. Folder-based dataset (with TB/Normal subfolders)"
    echo "3. CSV-based dataset (with image paths and labels)"
    echo
    read -p "Enter your choice (1-3): " format_choice
    
    case $format_choice in
        1) format="auto" ;;
        2) format="folders" ;;
        3) format="csv" ;;
        *) echo "Invalid choice. Using auto-detect."; format="auto" ;;
    esac
    
    echo
    echo "Converting dataset..."
    python convert_dataset.py --source_dir "$source_dir" --target_dir data --format $format
    echo
    echo "Dataset conversion completed! Images have been placed in data/train/normal, data/train/tb, and data/test directories."
    echo
    read -p "Press Enter to continue..."
    show_menu
}

function train {
    echo
    echo "=== Training the Model ==="
    echo
    echo "Choose training method:"
    echo "1. Train with Federated Learning (simulates training across 3 hospitals)"
    echo "2. Train without Federated Learning (centralized training)"
    echo
    read -p "Enter your choice (1-2): " train_choice
    
    case $train_choice in
        1) federated="true" ;;
        2) federated="false" ;;
        *) echo "Invalid choice. Using Federated Learning."; federated="true" ;;
    esac
    
    echo
    read -p "Enter number of federated rounds (default: 3): " rounds
    rounds=${rounds:-3}
    
    read -p "Enter number of epochs per round (default: 10): " epochs
    epochs=${epochs:-10}
    
    echo
    echo "Starting training..."
    python main.py train --federated $federated --rounds $rounds --epochs $epochs
    echo
    echo "Training completed!"
    echo
    read -p "Press Enter to continue..."
    show_menu
}

function predict {
    echo
    echo "=== Generating Predictions ==="
    echo
    echo "This will generate predictions for all images in the data/test directory."
    echo
    read -p "Number of test samples to visualize with GradCAM (default: 5): " visualize
    visualize=${visualize:-5}
    
    echo
    echo "Generating predictions..."
    python main.py predict --visualize_samples $visualize
    echo
    echo "Predictions completed!"
    echo "A submission.csv file has been created with the results."
    echo "GradCAM visualizations have been saved in the visualizations directory."
    echo
    read -p "Press Enter to continue..."
    show_menu
}

function streamlit {
    echo
    echo "=== Launching Web Interface ==="
    echo
    echo "This will start a Streamlit web application where you can:"
    echo "- Upload chest X-rays for TB detection"
    echo "- View model's prediction results"
    echo "- See GradCAM visualizations"
    echo
    echo "Press Ctrl+C in the terminal to stop the application when done."
    echo
    read -p "Press Enter to launch the web interface..."
    echo "Launching web interface..."
    python main.py streamlit
    show_menu
}

# Start the script
show_menu 