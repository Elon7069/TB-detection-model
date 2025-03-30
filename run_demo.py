#!/usr/bin/env python
"""
TB Detection Demo Runner

This script runs the TB Detection system in demo mode using a mock TensorFlow implementation.
It allows you to see how the system would work without needing to install TensorFlow.
"""

import os
import sys
import numpy as np
import argparse
import shutil

def setup_mock_tensorflow():
    """Set up mock TensorFlow module."""
    print("Setting up mock TensorFlow for demonstration...")
    
    # Create directory for mock modules
    os.makedirs("mock_modules", exist_ok=True)
    
    # Create __init__.py in mock_modules
    with open(os.path.join("mock_modules", "__init__.py"), "w") as f:
        f.write("# Mock modules package\n")
    
    # Copy mock_tensorflow.py to mock_modules/tensorflow.py
    shutil.copy("mock_tensorflow.py", os.path.join("mock_modules", "tensorflow.py"))
    
    # Add mock_modules to Python path
    sys.path.insert(0, os.path.abspath("mock_modules"))
    
    # Create a symlink to the mock tensorflow module
    try:
        import tensorflow
        print(f"Found existing TensorFlow: {tensorflow.__version__}")
    except ImportError:
        print("No TensorFlow found, using mock version.")
        
        # Import the mock tensorflow
        sys.modules["tensorflow"] = __import__("tensorflow")
        import tensorflow
        
        # Also make tensorflow.keras available
        sys.modules["tensorflow.keras"] = tensorflow.keras

def create_sample_data():
    """Create sample X-ray data for demo purposes."""
    print("Creating sample X-ray data for demonstration...")
    
    # Create directories
    os.makedirs("data/train/normal", exist_ok=True)
    os.makedirs("data/train/tb", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    os.makedirs("fl_models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Create 10 sample images in each directory
    for i in range(10):
        # Create a random 224x224 grayscale image
        img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        
        # Normal samples
        with open(f"data/train/normal/normal_{i:03d}.png", "wb") as f:
            f.write(b"PNG DEMO IMAGE")
            
        # TB samples
        with open(f"data/train/tb/tb_{i:03d}.png", "wb") as f:
            f.write(b"PNG DEMO IMAGE")
            
        # Test samples
        with open(f"data/test/TB_test_{i:04d}.png", "wb") as f:
            f.write(b"PNG DEMO IMAGE")
    
    print("Created sample data in data/train/normal, data/train/tb, and data/test directories.")

def run_demo():
    """Run the TB detection demo."""
    parser = argparse.ArgumentParser(description='TB Detection Demo')
    parser.add_argument('--mode', type=str, choices=['setup', 'train', 'predict', 'streamlit'], 
                        default='setup', help='Mode to run')
    args = parser.parse_args()
    
    # Set up mock TensorFlow
    setup_mock_tensorflow()
    
    # Create sample data
    if args.mode == 'setup':
        create_sample_data()
        print("\nSetup completed. You can now run:")
        print("  python run_demo.py --mode train")
        print("  python run_demo.py --mode predict")
        return
    
    # Import the main module
    print("\nRunning TB Detection in demo mode...")
    
    if args.mode == 'train':
        print("\nTraining model (demo mode)...")
        # Mock the training process
        for i in range(3):
            print(f"Epoch {i+1}/3")
            print(f"loss: 0.{9-i}234 - accuracy: 0.{i+7}123")
        
        # Create mock model files
        os.makedirs("fl_models", exist_ok=True)
        with open("fl_models/server_model_fine_tuned.h5", "w") as f:
            f.write("Mock model weights")
        
        print("\nTraining completed successfully!")
        print("You can now run:")
        print("  python run_demo.py --mode predict")
    
    elif args.mode == 'predict':
        print("\nGenerating predictions (demo mode)...")
        
        # Create mock submission file
        with open("submission.csv", "w") as f:
            f.write("ID,Target\n")
            for i in range(10):
                # Randomly assign 0 or 1 as the prediction
                pred = np.random.randint(0, 2)
                f.write(f"TB_test_{i:04d},{pred}\n")
        
        # Create mock visualizations
        os.makedirs("visualizations", exist_ok=True)
        print("Generating GradCAM visualizations...")
        for i in range(5):
            # Ensure the directory exists
            vis_file = f"visualizations/gradcam_TB_test_{i:04d}.png"
            os.makedirs(os.path.dirname(vis_file), exist_ok=True)
            
            # Write the file
            with open(vis_file, "w") as f:
                f.write("Mock GradCAM visualization")
        
        print("\nPrediction completed successfully!")
        print("Generated submission.csv and GradCAM visualizations in visualizations/ directory.")
        
        # Verify files were created
        try:
            grad_files = os.listdir("visualizations")
            print(f"Created {len(grad_files)} GradCAM visualization files.")
        except Exception as e:
            print(f"Warning: Could not verify visualizations: {str(e)}")
    
    elif args.mode == 'streamlit':
        print("\nLaunching Streamlit web app (demo mode)...")
        print("This would normally start the Streamlit interface.")
        print("In demo mode, we're just simulating the process.")
        print("\nStreamlit would allow you to:")
        print("1. Upload a chest X-ray image")
        print("2. Get a prediction (TB or Normal)")
        print("3. View GradCAM visualization of the model's focus areas")

if __name__ == "__main__":
    run_demo() 