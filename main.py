#!/usr/bin/env python
"""
TB Detection System - Main Script

This script runs the entire TB detection pipeline from data loading to model training,
evaluation, and visualization for the Techkriti 2025 ML Hackathon.
"""

import os
import argparse
import json
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Import project modules
from data_loader import TBDataset, create_dataloaders
from model import TBResNet50, initialize_model
from train import train_model, validate_model
from eval_metrics import analyze_performance
from ensemble import (
    EnsemblePredictor, optimize_ensemble_weights, 
    save_ensemble_config, visualize_ensemble_optimization
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TB Detection System')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save models and results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (L2 penalty)')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='Base model architecture (currently only resnet50 supported)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone layers')
    
    # Federated learning parameters
    parser.add_argument('--federated', action='store_true',
                        help='Use federated learning (split data into clients)')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of clients for federated learning')
    parser.add_argument('--federated_rounds', type=int, default=5,
                        help='Number of federated learning rounds')
    
    # Ensemble parameters
    parser.add_argument('--ensemble', action='store_true',
                        help='Create an ensemble of models')
    parser.add_argument('--num_models', type=int, default=3,
                        help='Number of models for ensemble')
    
    # Execution modes
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'ensemble', 'federated'],
                        help='Execution mode')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to model to load for evaluation')
    
    # Device parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, or None for auto-detection)')
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the execution environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"Using device: {args.device}")
    print(f"Configuration saved to {config_path}")
    
    return args

def train_single_model(args, train_loader, val_loader):
    """Train a single TB detection model."""
    # Initialize model
    model = initialize_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(args.device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Train model
    model, train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir
    )
    
    # Save model and metrics
    model_path = os.path.join(args.output_dir, 'tb_model.pt')
    torch.save(model, model_path)
    
    metrics_path = os.path.join(args.output_dir, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=4)
    
    print(f"Model saved to {model_path}")
    print(f"Training metrics saved to {metrics_path}")
    
    return model, train_metrics

def evaluate_model(args, model, test_loader):
    """Evaluate a trained model on the test set."""
    # Run evaluation
    y_true, y_pred_proba = validate_model(
        model=model,
        data_loader=test_loader,
        device=args.device,
        return_predictions=True
    )
    
    # Analyze performance
    performance_dir = os.path.join(args.output_dir, 'evaluation')
    performance = analyze_performance(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        save_dir=performance_dir
    )
    
    # Print performance summary
    print("\n=== TB Detection Performance Report ===")
    print(f"Optimal Threshold: {performance['optimal_threshold']:.2f}")
    print(f"F1-score: {performance['f1_score']:.4f} ({performance['baseline_comparison']['f1_status']})")
    print(f"Recall: {performance['recall']:.4f} ({performance['baseline_comparison']['recall_status']})")
    print(f"Precision: {performance['precision']:.4f}")
    print(f"Accuracy: {performance['accuracy']:.4f}")
    print(f"\nHackathon Competitiveness: {'✓' if performance['hackathon_competitiveness'] else '✗'}")
    print(f"Visualizations saved in '{performance_dir}' directory.")
    
    return performance

def train_ensemble(args, train_loader, val_loader, test_loader):
    """Train multiple models and create an ensemble."""
    model_paths = []
    
    # Train multiple models with different initializations
    ensemble_dir = os.path.join(args.output_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)
    
    for i in range(args.num_models):
        print(f"\n=== Training Model {i+1}/{args.num_models} for Ensemble ===")
        
        # Initialize with different random seeds
        torch.manual_seed(i)
        np.random.seed(i)
        
        # Model-specific output directory
        model_output_dir = os.path.join(ensemble_dir, f'model_{i+1}')
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Create a new args object with updated output_dir
        model_args = argparse.Namespace(**vars(args))
        model_args.output_dir = model_output_dir
        
        # Train the model
        model, _ = train_single_model(model_args, train_loader, val_loader)
        
        # Save the model
        model_path = os.path.join(model_output_dir, 'tb_model.pt')
        model_paths.append(model_path)
    
    # Get validation predictions for ensemble optimization
    val_preds = []
    val_labels = []
    
    # Create a dataset of validation set predictions from each model
    for model_path in model_paths:
        model = torch.load(model_path, map_location=args.device)
        model.eval()
        
        y_true, y_pred = validate_model(
            model=model,
            data_loader=val_loader,
            device=args.device,
            return_predictions=True
        )
        
        val_preds.append(y_pred)
        val_labels = y_true  # Will be the same for all models
    
    # Optimize ensemble weights
    print("\n=== Optimizing Ensemble Weights ===")
    best_weights, best_threshold, history = optimize_ensemble_weights(
        model_paths=model_paths,
        X_val=None,  # Not needed as we've pre-computed predictions
        y_val=val_labels,
        metric='hackathon',
        n_iterations=50
    )
    
    # Save ensemble configuration
    ensemble_config_path = os.path.join(ensemble_dir, 'ensemble_config.json')
    save_ensemble_config(
        output_path=ensemble_config_path,
        model_paths=model_paths,
        weights=best_weights,
        threshold=best_threshold,
        metrics={'optimized_for': 'hackathon_score'},
        device=args.device
    )
    
    # Visualize optimization
    visualization_path = os.path.join(ensemble_dir, 'ensemble_optimization.png')
    visualize_ensemble_optimization(history, visualization_path)
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(model_paths, best_weights)
    ensemble.load_models(device=args.device)
    
    # Evaluate ensemble on test set
    print("\n=== Evaluating Ensemble on Test Set ===")
    y_true, y_pred_proba = validate_model(
        model=model,  # Not used as we'll compute predictions manually
        data_loader=test_loader,
        device=args.device,
        return_predictions=True,
        ensemble_predictor=ensemble
    )
    
    # Analyze performance
    performance_dir = os.path.join(ensemble_dir, 'evaluation')
    performance = analyze_performance(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        save_dir=performance_dir
    )
    
    # Print performance summary
    print("\n=== Ensemble Performance Report ===")
    print(f"Optimal Threshold: {performance['optimal_threshold']:.2f}")
    print(f"F1-score: {performance['f1_score']:.4f} ({performance['baseline_comparison']['f1_status']})")
    print(f"Recall: {performance['recall']:.4f} ({performance['baseline_comparison']['recall_status']})")
    print(f"Precision: {performance['precision']:.4f}")
    print(f"Accuracy: {performance['accuracy']:.4f}")
    print(f"\nHackathon Competitiveness: {'✓' if performance['hackathon_competitiveness'] else '✗'}")
    print(f"Visualizations saved in '{performance_dir}' directory.")
    
    return ensemble, performance

def train_federated(args, train_dataset, val_loader, test_loader):
    """
    Implement federated learning by training models on different data partitions
    and then aggregating the models.
    """
    # Split training data into client partitions
    num_samples = len(train_dataset)
    samples_per_client = num_samples // args.num_clients
    
    # Create client datasets using random split
    indices = torch.randperm(num_samples).tolist()
    client_datasets = []
    
    for i in range(args.num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < args.num_clients - 1 else num_samples
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(torch.utils.data.Subset(train_dataset, client_indices))
    
    # Create client dataloaders
    client_loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
        for dataset in client_datasets
    ]
    
    # Create directory for federated learning results
    federated_dir = os.path.join(args.output_dir, 'federated')
    os.makedirs(federated_dir, exist_ok=True)
    
    # Initialize global model
    global_model = initialize_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(args.device)
    
    # Federated learning rounds
    round_metrics = []
    
    for round_num in range(args.federated_rounds):
        print(f"\n=== Federated Learning Round {round_num+1}/{args.federated_rounds} ===")
        
        # Initialize list to store client models
        client_models = []
        
        # Train models on each client
        for client_id, client_loader in enumerate(client_loaders):
            print(f"Training client {client_id+1}/{args.num_clients}")
            
            # Create a copy of the global model for this client
            client_model = type(global_model)()
            client_model.load_state_dict(global_model.state_dict())
            client_model = client_model.to(args.device)
            
            # Set up optimizer and loss function
            optimizer = torch.optim.Adam(
                client_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            
            criterion = torch.nn.BCEWithLogitsLoss()
            
            # Train client model for one epoch
            local_epochs = 1
            for epoch in range(local_epochs):
                client_model.train()
                running_loss = 0.0
                
                for inputs, labels in client_loader:
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device, dtype=torch.float32)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = client_model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
            
            client_models.append(client_model)
        
        # Aggregate client models (simple averaging of weights)
        with torch.no_grad():
            # Get state dictionaries of all clients
            client_states = [model.state_dict() for model in client_models]
            
            # Average the weights
            for key in global_model.state_dict().keys():
                # Skip batch norm statistics for now (more advanced methods would handle these differently)
                if 'running_mean' in key or 'running_var' in key:
                    continue
                    
                # Average weights from all clients
                global_model.state_dict()[key] = torch.stack(
                    [client_states[i][key] for i in range(args.num_clients)]
                ).mean(0)
        
        # Evaluate global model after aggregation
        val_metrics = validate_model(
            model=global_model,
            data_loader=val_loader,
            device=args.device
        )
        
        # Log metrics for this round
        round_metrics.append({
            'round': round_num + 1,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'val_recall': val_metrics['recall']
        })
        
        print(f"Round {round_num+1} - Validation: Loss={val_metrics['loss']:.4f}, "
              f"Accuracy={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
              f"Recall={val_metrics['recall']:.4f}")
    
    # Save the final global model
    model_path = os.path.join(federated_dir, 'global_model.pt')
    torch.save(global_model, model_path)
    
    # Save federated learning metrics
    metrics_path = os.path.join(federated_dir, 'federated_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(round_metrics, f, indent=4)
    
    # Evaluate final global model on test set
    performance = evaluate_model(args, global_model, test_loader)
    
    return global_model, performance

def main():
    """Main function to run the TB detection system."""
    # Parse arguments
    args = parse_args()
    args = setup_environment(args)
    
    # Create data loaders
    train_dataset, train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Execute based on mode
    if args.mode == 'train':
        # Train a single model
        model, _ = train_single_model(args, train_loader, val_loader)
        
        # Evaluate on test set
        evaluate_model(args, model, test_loader)
    
    elif args.mode == 'eval':
        # Load model for evaluation
        if args.load_model is None:
            model_path = os.path.join(args.output_dir, 'tb_model.pt')
        else:
            model_path = args.load_model
        
        print(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=args.device)
        
        # Evaluate on test set
        evaluate_model(args, model, test_loader)
    
    elif args.mode == 'ensemble':
        # Train and evaluate ensemble
        train_ensemble(args, train_loader, val_loader, test_loader)
    
    elif args.mode == 'federated':
        # Train using federated learning
        train_federated(args, train_dataset, val_loader, test_loader)
    
    print("\nTB Detection System execution completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds") 