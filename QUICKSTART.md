# TB Detection System - Quick Start Guide

This guide provides the fastest way to get started with the TB detection system built for the Techkriti 2025 ML Hackathon.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**:
   Place your chest X-ray images in the following structure:
   ```
   data/
   ├── train/
   │   ├── normal/        # Normal chest X-ray images 
   │   └── tuberculosis/  # TB chest X-ray images
   ├── val/ (optional - will be created from train if not present)
   │   ├── normal/
   │   └── tuberculosis/
   └── test/
       ├── normal/
       └── tuberculosis/
   ```

   If you only have a single set of images, you can use our data splitting utility:
   ```bash
   python data_loader.py --source_dir your_images_dir --output_dir data --split 0.7 0.15 0.15
   ```

## Training a Model

**Basic training**:
```bash
python main.py --mode train --data_dir data --output_dir output --pretrained
```

This will:
- Load the dataset from the `data` directory
- Initialize a pre-trained ResNet50 model
- Train for 10 epochs (default)
- Save the model and training metrics to the `output` directory
- Evaluate the model on the test set
- Generate visualizations and performance metrics

## Evaluation

To evaluate a trained model:
```bash
python main.py --mode eval --load_model output/tb_model.pt
```

## Advanced Methods

### Ensemble Learning
```bash
python main.py --mode ensemble --num_models 3 --num_epochs 15
```

### Federated Learning
```bash
python main.py --mode federated --num_clients 3 --federated_rounds 10
```

## Results

After training or evaluation, check the output directory for:
- Trained model file(s)
- Training metrics (JSON format)
- Performance visualizations:
  - `evaluation/confusion_matrix.png`
  - `evaluation/hackathon_metrics.png`
  - `evaluation/performance_report.json`

## Common Issues & Solutions

1. **CUDA out of memory error**:
   - Reduce batch size: `--batch_size 8`
   - Use a smaller model or freeze more layers: `--freeze_backbone`

2. **Overfitting**:
   - Increase weight decay: `--weight_decay 0.001`
   - Reduce number of epochs: `--num_epochs 5`

3. **Poor performance**:
   - Ensure your dataset is balanced
   - Try ensemble mode for better results
   - Increase learning rate: `--learning_rate 0.005`

## Example Command Sequence

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a basic model
python main.py --mode train --pretrained --num_epochs 15

# 3. Train an ensemble for better performance
python main.py --mode ensemble --num_models 3 --num_epochs 10

# 4. Compare results
python main.py --mode eval --load_model output/tb_model.pt
python main.py --mode eval --load_model output/ensemble/ensemble_config.json
``` 