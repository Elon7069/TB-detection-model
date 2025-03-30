#!/usr/bin/env python
"""
Ensemble Modeling for TB Detection

This script implements various ensemble techniques to combine predictions
from multiple TB detection models to achieve better performance.
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple, Union, Optional

class EnsemblePredictor:
    """
    Class for performing ensemble predictions by combining
    outputs from multiple models.
    """
    
    def __init__(self, model_paths: List[str], weights: Optional[List[float]] = None):
        """
        Initialize the ensemble predictor.
        
        Args:
            model_paths: List of paths to saved models or predictions
            weights: Optional weights for each model (default: equal weights)
        """
        self.model_paths = model_paths
        self.num_models = len(model_paths)
        
        # Set weights
        if weights is None:
            self.weights = np.ones(self.num_models) / self.num_models
        else:
            # Normalize weights to sum to 1
            self.weights = np.array(weights) / sum(weights)
        
        # Validate
        if len(self.weights) != self.num_models:
            raise ValueError("Number of weights must match number of models")
            
        self.models = []
        self.model_types = []
        
    def load_models(self, device='cuda'):
        """
        Load all models in the ensemble.
        
        Args:
            device: Device to load PyTorch models to ('cuda' or 'cpu')
        """
        self.models = []
        self.model_types = []
        
        for path in self.model_paths:
            if path.endswith('.pt') or path.endswith('.pth'):
                # PyTorch model
                model = torch.load(path, map_location=device)
                model.eval()
                self.model_types.append('pytorch')
            elif path.endswith('.pkl'):
                # Scikit-learn model
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                self.model_types.append('sklearn')
            elif path.endswith('.npy'):
                # Pre-computed predictions
                model = np.load(path)
                self.model_types.append('predictions')
            else:
                raise ValueError(f"Unsupported model format: {path}")
            
            self.models.append(model)
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Generate ensemble prediction probabilities.
        
        Args:
            X: Input data (numpy array or PyTorch tensor)
            
        Returns:
            Array of prediction probabilities [samples, classes]
        """
        all_predictions = []
        
        for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
            if model_type == 'pytorch':
                # PyTorch model
                if not isinstance(X, torch.Tensor):
                    # Convert to PyTorch tensor if needed
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                else:
                    X_tensor = X
                
                with torch.no_grad():
                    outputs = model(X_tensor)
                    # Get probabilities
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Some models return (outputs, features)
                    probs = torch.sigmoid(outputs).cpu().numpy()
            
            elif model_type == 'sklearn':
                # Scikit-learn model
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    # If binary classification, take the positive class probability
                    if probs.shape[1] == 2:
                        probs = probs[:, 1]
                else:
                    # Use decision function if predict_proba not available
                    probs = model.decision_function(X)
                    # Convert to probability-like values between 0 and 1
                    probs = 1 / (1 + np.exp(-probs))
            
            elif model_type == 'predictions':
                # Pre-computed predictions
                probs = model
            
            # Ensure probs is 2D [samples, 1] for consistent handling
            if len(probs.shape) == 1:
                probs = probs.reshape(-1, 1)
            
            # Append predictions with their weight
            all_predictions.append(probs * self.weights[i])
        
        # Sum weighted predictions
        ensemble_probs = sum(all_predictions)
        
        # If binary classification (TB detection case)
        if ensemble_probs.shape[1] == 1:
            return ensemble_probs.flatten()
        else:
            return ensemble_probs
    
    def predict(self, X: Union[np.ndarray, torch.Tensor], threshold=0.5) -> np.ndarray:
        """
        Generate ensemble class predictions.
        
        Args:
            X: Input data
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Array of class predictions
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

def optimize_ensemble_weights(
    model_paths: List[str],
    X_val: Union[np.ndarray, torch.Tensor],
    y_val: np.ndarray,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    metric: str = 'f1',
    val_domains: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[List[float], Dict]:
    """
    Optimize the weights of ensemble models using gradient descent
    to maximize a given metric, with additional considerations for domain generalization.
    
    Args:
        model_paths: List of paths to model files
        X_val: Validation data
        y_val: Validation labels
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for gradient descent
        metric: Metric to optimize ('f1', 'recall', or 'hackathon')
        val_domains: Optional dictionary of validation data from different domains
        
    Returns:
        Tuple of (optimized_weights, optimal_threshold, metrics_history)
    """
    # Initialize ensemble with equal weights
    n_models = len(model_paths)
    weights = np.ones(n_models) / n_models
    
    # Initialize ensemble
    ensemble = EnsemblePredictor(model_paths, weights)
    ensemble.load_models()
    
    # Initialize metrics history
    history = {
        'weights': [],
        metric: [],
        'thresholds': [],
        'domain_metrics': [] if val_domains else None
    }
    
    # Get base predictions from each model
    base_preds = []
    for i in range(n_models):
        # Set weight for only one model
        temp_weights = np.zeros(n_models)
        temp_weights[i] = 1.0
        ensemble.weights = temp_weights
        
        # Get predictions
        preds = ensemble.predict_proba(X_val)
        base_preds.append(preds)
    
    base_preds = np.array(base_preds)
    
    # If we have domain validation data, also get predictions for each domain
    domain_base_preds = {}
    if val_domains:
        for domain_name, X_domain in val_domains.items():
            domain_preds = []
            for i in range(n_models):
                temp_weights = np.zeros(n_models)
                temp_weights[i] = 1.0
                ensemble.weights = temp_weights
                
                domain_preds.append(ensemble.predict_proba(X_domain))
            
            domain_base_preds[domain_name] = np.array(domain_preds)
    
    # Optimization loop
    best_score = 0
    best_weights = weights.copy()
    best_threshold = 0.5
    
    for iteration in range(n_iterations):
        # Calculate weighted sum of predictions
        ensemble_preds = np.zeros_like(y_val, dtype=float)
        for i in range(n_models):
            ensemble_preds += weights[i] * base_preds[i]
        
        # Find optimal threshold
        thresholds = np.linspace(0.1, 0.9, 41)
        best_iter_score = 0
        domain_scores = {}
        
        for threshold in thresholds:
            y_pred = (ensemble_preds > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            elif metric == 'hackathon':
                # F1 with recall tiebreaker
                f1 = f1_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                score = f1 + 0.01 * recall
            elif metric == 'generalization':
                # Base score is still F1 with recall tiebreaker
                f1 = f1_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                base_score = f1 + 0.01 * recall
                
                # But if we have domain data, we also consider cross-domain performance
                if val_domains:
                    domain_metrics = {}
                    for domain_name, domain_preds_list in domain_base_preds.items():
                        # Calculate domain predictions with current weights
                        domain_ensemble_preds = np.zeros_like(y_val, dtype=float)
                        for i in range(n_models):
                            domain_ensemble_preds += weights[i] * domain_preds_list[i]
                        
                        # Apply threshold
                        domain_y_pred = (domain_ensemble_preds > threshold).astype(int)
                        
                        # Calculate domain metrics
                        domain_f1 = f1_score(y_val, domain_y_pred)
                        domain_recall = recall_score(y_val, domain_y_pred)
                        domain_score = domain_f1 + 0.01 * domain_recall
                        
                        domain_metrics[domain_name] = {
                            'f1': domain_f1,
                            'recall': domain_recall,
                            'score': domain_score
                        }
                    
                    # Calculate generalization score: base_score * 0.6 + min_domain_score * 0.4
                    # This prioritizes models that perform well even on the most challenging domains
                    min_domain_score = min(domain_metrics[d]['score'] for d in domain_metrics)
                    score = base_score * 0.6 + min_domain_score * 0.4
                    
                    # Save domain metrics for this threshold if it's the best so far
                    if score > best_iter_score:
                        domain_scores = domain_metrics
                else:
                    score = base_score
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_iter_score:
                best_iter_score = score
                best_iter_threshold = threshold
        
        # Update best overall
        if best_iter_score > best_score:
            best_score = best_iter_score
            best_weights = weights.copy()
            best_threshold = best_iter_threshold
        
        # Calculate gradient and update weights
        weight_grads = np.zeros(n_models)
        
        for i in range(n_models):
            # Small perturbation for each weight
            perturbed_weights = weights.copy()
            perturbed_weights[i] += 0.01
            perturbed_weights = perturbed_weights / np.sum(perturbed_weights)
            
            # Calculate predictions with perturbed weights
            perturbed_preds = np.zeros_like(y_val, dtype=float)
            for j in range(n_models):
                perturbed_preds += perturbed_weights[j] * base_preds[j]
            
            # Calculate score with perturbed weights
            y_pred = (perturbed_preds > best_iter_threshold).astype(int)
            
            if metric == 'f1':
                perturbed_score = f1_score(y_val, y_pred)
            elif metric == 'recall':
                perturbed_score = recall_score(y_val, y_pred)
            elif metric == 'hackathon':
                f1 = f1_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                perturbed_score = f1 + 0.01 * recall
            elif metric == 'generalization':
                # Base score with perturbed weights
                f1 = f1_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                base_score = f1 + 0.01 * recall
                
                # Also evaluate on domain data if available
                if val_domains:
                    domain_scores_perturbed = []
                    for domain_name, domain_preds_list in domain_base_preds.items():
                        # Calculate domain predictions with perturbed weights
                        domain_perturbed_preds = np.zeros_like(y_val, dtype=float)
                        for j in range(n_models):
                            domain_perturbed_preds += perturbed_weights[j] * domain_preds_list[j]
                        
                        # Apply threshold
                        domain_y_pred = (domain_perturbed_preds > best_iter_threshold).astype(int)
                        
                        # Calculate domain score
                        domain_f1 = f1_score(y_val, domain_y_pred)
                        domain_recall = recall_score(y_val, domain_y_pred)
                        domain_score = domain_f1 + 0.01 * domain_recall
                        domain_scores_perturbed.append(domain_score)
                    
                    # Calculate generalization score: base_score * 0.6 + min_domain_score * 0.4
                    min_domain_score = min(domain_scores_perturbed)
                    perturbed_score = base_score * 0.6 + min_domain_score * 0.4
                else:
                    perturbed_score = base_score
            
            # Calculate gradient (score improvement / weight change)
            weight_grads[i] = (perturbed_score - best_iter_score) / 0.01
        
        # Update weights
        weights += learning_rate * weight_grads
        
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        # Record history
        history['weights'].append(weights.copy())
        history[metric].append(best_iter_score)
        history['thresholds'].append(best_iter_threshold)
        
        # Record domain metrics if available
        if val_domains and history['domain_metrics'] is not None:
            history['domain_metrics'].append(domain_scores)
        
        # Early stopping if little improvement
        if iteration > 10 and np.mean(np.diff(history[metric][-10:])) < 0.0001:
            break
    
    return best_weights, best_threshold, history

def save_ensemble_config(
    output_path: str,
    model_paths: List[str],
    weights: List[float],
    threshold: float,
    metrics: Dict,
    device: str = 'cpu'
):
    """
    Save ensemble configuration and results.
    
    Args:
        output_path: Path to save ensemble configuration
        model_paths: List of model paths
        weights: Optimized model weights
        threshold: Optimized threshold
        metrics: Metrics from validation
        device: Device for inference
    """
    config = {
        'model_paths': model_paths,
        'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
        'threshold': float(threshold),
        'metrics': metrics,
        'device': device
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_ensemble_config(config_path: str) -> EnsemblePredictor:
    """
    Load ensemble configuration and create an ensemble predictor.
    
    Args:
        config_path: Path to ensemble configuration
        
    Returns:
        Configured EnsemblePredictor
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    ensemble = EnsemblePredictor(
        config['model_paths'],
        config['weights']
    )
    
    ensemble.load_models(device=config.get('device', 'cpu'))
    
    return ensemble, config['threshold']

def visualize_ensemble_optimization(history: Dict, save_path: Optional[str] = None):
    """
    Visualize ensemble optimization process.
    
    Args:
        history: Optimization history dictionary
        save_path: Path to save visualization
    """
    metric_name = next(key for key in history.keys() if key not in ['weights', 'thresholds'])
    
    weights_history = np.array(history['weights'])
    metric_history = np.array(history[metric_name])
    threshold_history = np.array(history['thresholds'])
    
    n_models = weights_history.shape[1]
    n_iterations = len(metric_history)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot metric
    ax1.plot(range(n_iterations), metric_history, 'b-', linewidth=2)
    ax1.set_ylabel(f'{metric_name.upper()} Score', fontsize=12)
    ax1.set_title(f'Ensemble Optimization - {metric_name.upper()} Score', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot thresholds
    ax2.plot(range(n_iterations), threshold_history, 'g-', linewidth=2)
    ax2.set_ylabel('Threshold', fontsize=12)
    ax2.set_title('Optimal Threshold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot weights
    for i in range(n_models):
        ax3.plot(range(n_iterations), weights_history[:, i], 
                linewidth=2, label=f'Model {i+1}')
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Weight', fontsize=12)
    ax3.set_title('Model Weights', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def evaluate_ensemble_generalization(
    ensemble: EnsemblePredictor,
    threshold: float,
    X_test_domains: Dict[str, np.ndarray],
    y_test: np.ndarray,
    output_dir: str = 'ensemble_generalization'
) -> Dict:
    """
    Evaluate the ensemble model's generalization capabilities across different domains.
    
    Args:
        ensemble: Configured ensemble predictor
        threshold: Classification threshold
        X_test_domains: Dictionary of test data from different domains
        y_test: Test labels
        output_dir: Directory to save results
        
    Returns:
        Dictionary of generalization metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Results for each domain
    domain_results = {}
    
    # Evaluate on each domain
    for domain_name, X_domain in X_test_domains.items():
        print(f"\nEvaluating ensemble on domain: {domain_name}")
        
        # Get predictions
        y_pred_proba = ensemble.predict_proba(X_domain)
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'f1': f1_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'accuracy': np.mean(y_test == y_pred)
        }
        
        domain_results[domain_name] = metrics
        print(f"F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    # Calculate generalization metrics
    f1_values = [domain_results[d]['f1'] for d in domain_results]
    recall_values = [domain_results[d]['recall'] for d in domain_results]
    
    generalization_metrics = {
        'min_f1': min(f1_values),
        'min_recall': min(recall_values),
        'f1_std': np.std(f1_values),
        'recall_std': np.std(recall_values),
        'f1_mean': np.mean(f1_values),
        'recall_mean': np.mean(recall_values),
        'domain_results': domain_results
    }
    
    # Save results
    with open(f'{output_dir}/generalization_metrics.json', 'w') as f:
        json.dump(generalization_metrics, f, indent=4)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    domains = list(domain_results.keys())
    x = np.arange(len(domains))
    width = 0.3
    
    plt.bar(x - width/2, [domain_results[d]['f1'] for d in domains], width, label='F1-score')
    plt.bar(x + width/2, [domain_results[d]['recall'] for d in domains], width, label='Recall')
    
    plt.axhline(y=generalization_metrics['f1_mean'], color='b', linestyle='--', 
               label=f"Avg F1: {generalization_metrics['f1_mean']:.3f}")
    plt.axhline(y=generalization_metrics['recall_mean'], color='orange', linestyle='--', 
               label=f"Avg Recall: {generalization_metrics['recall_mean']:.3f}")
    
    plt.title('Ensemble Generalization Performance Across Domains', fontsize=14)
    plt.xlabel('Domain', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, domains)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/domain_performance.png', dpi=300)
    plt.close()
    
    # Print summary
    print("\n=== Ensemble Generalization Assessment ===")
    print(f"F1-score: mean={generalization_metrics['f1_mean']:.4f}, "
          f"min={generalization_metrics['min_f1']:.4f}, std={generalization_metrics['f1_std']:.4f}")
    print(f"Recall: mean={generalization_metrics['recall_mean']:.4f}, "
          f"min={generalization_metrics['min_recall']:.4f}, std={generalization_metrics['recall_std']:.4f}")
    
    # Assess generalization quality
    if generalization_metrics['f1_std'] < 0.03 and generalization_metrics['recall_std'] < 0.03:
        print("✓ Excellent generalization: Consistent performance across all domains")
    elif generalization_metrics['f1_std'] < 0.07 and generalization_metrics['recall_std'] < 0.07:
        print("✓ Good generalization: Relatively stable performance across domains")
    else:
        print("! Poor generalization: Performance varies significantly across domains")
    
    return generalization_metrics

def main():
    """
    Example usage of ensemble functionality.
    """
    # For demonstration: Generate synthetic models and predictions
    n_samples = 100
    n_models = 3
    
    # Create synthetic validation data
    np.random.seed(42)
    X_val = np.random.randn(n_samples, 10)
    y_val = np.random.randint(0, 2, n_samples)
    
    # Create synthetic model predictions
    model_preds = []
    for i in range(n_models):
        # Generate predictions with different levels of accuracy
        base_preds = np.random.random(n_samples)
        # Bias toward correct labels with varying strength
        bias = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7
        preds = base_preds * (1 - bias) + y_val * bias
        
        # Save predictions
        pred_path = f'model_{i+1}_preds.npy'
        np.save(pred_path, preds)
        model_preds.append(pred_path)
    
    # Optimize ensemble
    print("Optimizing ensemble weights...")
    best_weights, best_threshold, history = optimize_ensemble_weights(
        model_preds, X_val, y_val, metric='hackathon', n_iterations=50
    )
    
    # Create and evaluate final ensemble
    ensemble = EnsemblePredictor(model_preds, best_weights)
    ensemble.load_models()
    
    # Make predictions with optimal threshold
    y_pred = ensemble.predict(X_val, threshold=best_threshold)
    
    # Calculate metrics
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print("\n=== Ensemble Results ===")
    print(f"Optimal Weights: {best_weights}")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save configuration
    save_ensemble_config(
        'ensemble_config.json',
        model_preds,
        best_weights,
        best_threshold,
        {'f1': f1, 'recall': recall}
    )
    
    # Visualize optimization
    visualize_ensemble_optimization(history, 'ensemble_optimization.png')
    print("Ensemble configuration saved and visualization created.")
    
    # Clean up files created for demonstration
    for path in model_preds:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main() 