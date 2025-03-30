# TB Detection System

A comprehensive system for tuberculosis (TB) detection from chest X-ray images, developed for the Techkriti 2025 ML Hackathon.

## Overview

This system provides an end-to-end solution for detecting tuberculosis from chest X-ray images using deep learning. It includes data preprocessing, model training with ResNet50, performance evaluation, and visualization. The system supports multiple advanced techniques:

1. **Transfer Learning** with pre-trained ResNet50
2. **Federated Learning** for privacy-preserving distributed training
3. **Ensemble Learning** to improve prediction accuracy
4. **Threshold Optimization** targeting hackathon metrics (F1-score with recall tiebreaker)
5. **Cross-Domain Generalization** with robust evaluation across different X-ray qualities

## Enhanced Generalization Capabilities

Our system incorporates advanced techniques for ensuring that the model generalizes well across diverse cases:

### Data Augmentation & Domain Simulation
- **Tiered Augmentation Levels**: Light, medium, and strong augmentation pipelines to simulate diverse imaging conditions
- **Hospital Domain Simulation**: Simulates X-rays from different hospitals with varying equipment qualities
- **X-ray Quality Variations**: Generates variations in contrast, brightness, noise, blur, and resolution
- **Realistic Artifacts**: Simulates occlusions, positioning variations, and equipment-specific patterns

### Robustness Training
- **Stratified Sampling**: Groups images by characteristics (brightness, contrast, detail) for balanced training
- **Characteristic-Aware Federated Learning**: Distributes image types realistically across federated clients
- **Domain-Focused Ensemble Optimization**: Optimizes ensemble models to perform well even on challenging domains

### Evaluation Across Domains
- **Cross-Domain Testing**: Evaluates models on multiple simulated domains to measure generalization
- **Generalization Metrics**: Reports variation (std), minimum performance, and weighted averages across domains
- **Visual Assessment**: Generates Grad-CAM visualizations across domains to verify consistent attention to relevant areas

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/tb-detection.git
cd tb-detection

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Organize your data in the following structure:

```
data/
├── train/
│   ├── normal/        # Normal chest X-ray images 
│   └── tuberculosis/  # TB chest X-ray images
├── val/
│   ├── normal/
│   └── tuberculosis/
└── test/
    ├── normal/
    └── tuberculosis/
```

You can use the data_loader.py module to automatically split your data if it's not already divided.

## Usage

The system provides different modes of operation through the main.py script:

### Basic Training

```bash
# Train a basic model with default parameters
python main.py --mode train --data_dir data --output_dir output --pretrained

# Use specific hyperparameters
python main.py --mode train --batch_size 16 --num_epochs 20 --learning_rate 0.0005
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --mode eval --load_model output/tb_model.pt
```

### Ensemble Training

```bash
# Train an ensemble of 3 models
python main.py --mode ensemble --num_models 3 --num_epochs 15
```

### Federated Learning

```bash
# Train with federated learning (3 clients)
python main.py --mode federated --num_clients 3 --federated_rounds 10
```

### Generalization Evaluation

```bash
# Evaluate model generalization across domains
python train_and_predict.py --mode evaluate_generalization --model_path tb_model_best.h5

# Train with different augmentation strengths
python train_and_predict.py --mode train --augmentation_strength strong

# Test ensemble generalization
python main.py --mode ensemble --evaluate_domains True
```

## Results Visualization

After training or evaluation, the system generates visualizations in the specified output directory:

- **Confusion Matrix**: Shows true positives, false positives, etc.
- **ROC Curve**: Receiver Operating Characteristic curve with AUC
- **Precision-Recall Curve**: Shows precision vs. recall with AUC
- **Performance Metrics**: F1-score, recall, precision, and accuracy at different thresholds
- **Ensemble Optimization**: Visualization of weight optimization
- **Domain Generalization**: Performance across different simulated domains
- **GradCAM Visualizations**: Attention maps for each domain to verify model focus

## System Components

- **data_loader.py**: Data loading, preprocessing, and domain simulation
- **model.py**: Definition of TB detection model (ResNet50 with customization)
- **train.py**: Training and validation functions
- **eval_metrics.py**: Performance evaluation tools and visualizations
- **ensemble.py**: Ensemble learning with cross-domain optimization
- **federated_learning.py**: Privacy-preserving distributed learning implementation
- **main.py**: Main script to run the entire pipeline

## Hackathon Metrics

This system optimizes for the Techkriti 2025 ML Hackathon metrics:
- Primary metric: F1-score
- Tiebreaker: Recall
- Target performance: F1-score > 0.90, Recall > 0.93
- Emphasis on generalization across diverse patient data

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See requirements.txt for all dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Montgomery County X-ray Set](https://openi.nlm.nih.gov/faq) - X-ray dataset source
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [torchvision](https://pytorch.org/vision/) - Computer vision package with pre-trained models
- [Flower](https://flower.dev/) - Federated learning framework 