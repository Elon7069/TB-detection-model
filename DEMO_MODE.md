# TB Detection Demo Mode

If you're having issues installing TensorFlow, particularly on Windows, you can run the TB Detection system in demo mode. This allows you to experience the workflow and understand how the system works without needing to install TensorFlow.

## Running in Demo Mode

The demo mode uses a mock TensorFlow implementation that simulates the behavior of the real TensorFlow. It creates sample data, simulates training, and generates mock predictions and visualizations.

To run in demo mode:

1. **Setup the demo environment**:
   ```
   python run_demo.py --mode setup
   ```
   This will create sample X-ray data in the data directory structure.

2. **Train the model (simulated)**:
   ```
   python run_demo.py --mode train
   ```
   This will simulate the training process and create mock model files.

3. **Generate predictions (simulated)**:
   ```
   python run_demo.py --mode predict
   ```
   This will create a mock submission.csv file and GradCAM visualizations.

4. **Launch the web interface (simulated)**:
   ```
   python run_demo.py --mode streamlit
   ```
   This will simulate launching the Streamlit interface.

## How It Works

The demo mode works by:

1. Creating a mock TensorFlow module that mimics the API of the real TensorFlow
2. Setting up sample data directories with placeholder files
3. Simulating the training and prediction processes
4. Generating realistic output files

## Files Created by Demo Mode

- `mock_modules/tensorflow.py` - Mock TensorFlow implementation
- `data/train/normal/*.png` - Sample normal X-ray placeholder files
- `data/train/tb/*.png` - Sample TB X-ray placeholder files
- `data/test/*.png` - Sample test X-ray placeholder files
- `fl_models/server_model_fine_tuned.h5` - Mock model file
- `submission.csv` - Mock predictions
- `visualizations/gradcam_*.png` - Mock GradCAM visualizations

## Transitioning to Real Mode

Once you have successfully installed TensorFlow, you can transition to the real mode by:

1. Deleting the `mock_modules` directory
2. Running the regular commands:
   ```
   python main.py setup
   python main.py train
   python main.py predict
   python main.py streamlit
   ```

## Installation Tips for Windows

If you want to install the real TensorFlow on Windows:

1. Use a specific Python version that is compatible (e.g., Python 3.8-3.10)
2. Install TensorFlow CPU version:
   ```
   pip install tensorflow==2.10.0
   ```
   
3. Or try TensorFlow with DirectML support for better GPU performance on Windows:
   ```
   pip install tensorflow==2.10.0 tensorflow-directml-plugin==0.4.0
   ```

## Troubleshooting

If you encounter issues with the demo mode:

1. Make sure you have NumPy installed:
   ```
   pip install numpy
   ```
   
2. Check that you have write permissions in the current directory
3. If you get module import errors, try running:
   ```
   pip install scikit-learn matplotlib pandas
   ``` 