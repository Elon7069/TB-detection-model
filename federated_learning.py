import numpy as np
import copy
import os
from tb_model import TBModel

class FederatedLearning:
    """
    Federated Learning implementation for TB detection.
    
    This class simulates a federated learning environment with multiple clients
    (hospitals) and a central server for model aggregation.
    """
    
    def __init__(self, num_clients=3, input_shape=(224, 224, 3)):
        """
        Initialize Federated Learning.
        
        Args:
            num_clients: Number of clients (hospitals)
            input_shape: Input shape of the images
        """
        self.num_clients = num_clients
        self.input_shape = input_shape
        
        # Initialize server model
        self.server_model = TBModel(input_shape=input_shape)
        
        # Initialize client models
        self.client_models = [TBModel(input_shape=input_shape) for _ in range(num_clients)]
        
        # Create output directory for FL models
        os.makedirs('fl_models', exist_ok=True)
    
    def train_round(self, client_data, epochs=5, batch_size=32):
        """
        Train one round of federated learning.
        
        Args:
            client_data: List of (X_train, y_train, X_val, y_val) tuples for each client
            epochs: Number of local training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics from all clients
        """
        client_metrics = {}
        client_weights = []  # Store client weights for weighted aggregation
        
        # Train each client model
        for i, ((X_train, y_train, X_val, y_val), model) in enumerate(zip(client_data, self.client_models)):
            print(f"\n=== Training Client {i+1}/{self.num_clients} ===")
            
            # Set client model weights from server model
            client_model_weights = self.server_model.model.get_weights()
            model.model.set_weights(client_model_weights)
            
            # Train the client model
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate on validation data
            val_metrics = model.evaluate(X_val, y_val)
            client_metrics[f'client_{i+1}'] = val_metrics
            
            # Calculate client weight for aggregation based on F1 score and data size
            # This prioritizes clients with better performance and more training data
            if 'f1' in val_metrics:
                # Use F1 score as the weight
                performance_weight = val_metrics['f1']
                
                # Also consider data size weight
                data_weight = len(X_train) / sum(len(data[0]) for data in client_data)
                
                # Combine weights (70% performance, 30% data size)
                client_weight = (0.7 * performance_weight) + (0.3 * data_weight)
            else:
                # Fallback to equal weighting
                client_weight = 1.0
                
            # Early stopping for underperforming clients
            if 'f1' in val_metrics and val_metrics['f1'] < 0.6:
                print(f"Client {i+1} is underperforming (F1: {val_metrics['f1']:.4f}).")
                print(f"Reducing influence in aggregation.")
                client_weight *= 0.5  # Reduce weight for poor performers
            
            client_weights.append(client_weight)
            
            # Save client model
            model.save_weights(f'fl_models/client_{i+1}_model.h5')
        
        # Normalize client weights
        if sum(client_weights) > 0:
            client_weights = [w / sum(client_weights) for w in client_weights]
            print(f"Client weights for aggregation: {[round(w, 3) for w in client_weights]}")
        else:
            # Equal weights if all clients failed
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Aggregate models (Weighted FedAvg algorithm)
        self._aggregate_models(client_weights)
        
        # Save server model
        self.server_model.save_weights('fl_models/server_model.h5')
        
        return client_metrics
    
    def _aggregate_models(self, client_weights=None):
        """
        Aggregate client models using weighted FedAvg algorithm.
        
        Args:
            client_weights: Optional list of weights for each client model
                           (default: equal weighting)
        """
        # Set default weights if not provided
        if client_weights is None:
            client_weights = [1.0 / self.num_clients] * self.num_clients
        
        # Get all client model weights
        client_weights_list = [model.model.get_weights() for model in self.client_models]
        
        # Initialize new weights with the same shape as client weights
        new_weights = [np.zeros_like(w) for w in client_weights_list[0]]
        
        # Weighted average of the weights
        for idx, (client_w, weight) in enumerate(zip(client_weights_list, client_weights)):
            print(f"Aggregating client {idx+1} with weight {weight:.4f}")
            for i, w in enumerate(client_w):
                new_weights[i] += w * weight
        
        # Update server model with new weights
        self.server_model.model.set_weights(new_weights)
        
        # Print aggregation summary
        print(f"Aggregated {self.num_clients} client models with weighted FedAvg")
    
    def get_final_model(self):
        """
        Get the final aggregated model.
        
        Returns:
            The final server model
        """
        return self.server_model
    
    def compare_client_models(self, X_test, y_test):
        """
        Compare the performance of individual client models and the aggregated model.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        # Evaluate each client model
        for i, model in enumerate(self.client_models):
            print(f"\n=== Evaluating Client {i+1} Model ===")
            metrics = model.evaluate(X_test, y_test)
            comparison[f'client_{i+1}'] = metrics
        
        # Evaluate the aggregated server model
        print("\n=== Evaluating Aggregated Model ===")
        metrics = self.server_model.evaluate(X_test, y_test)
        comparison['aggregated'] = metrics
        
        # Print comparison summary
        print("\n=== Model Comparison Summary ===")
        print("F1 Scores:")
        for name, metrics in comparison.items():
            print(f"{name}: {metrics.get('f1', 'N/A'):.4f}")
        
        return comparison
    
    def fine_tune_final_model(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Fine-tune the final aggregated model with all the training data.
        
        Args:
            X_train: Combined training data
            y_train: Combined training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of fine-tuning epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        print("\n=== Fine-tuning the Aggregated Model ===")
        
        # Apply progressive unfreezing for better fine-tuning
        # First train with frozen base layers
        print("Stage 1: Training with frozen base layers...")
        history1 = self.server_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs // 2,  # First half of epochs
            batch_size=batch_size
        )
        
        # Then unfreeze some layers and train with a lower learning rate
        print("Stage 2: Fine-tuning with unfrozen layers...")
        self.server_model.unfreeze_layers(num_layers=30)
        
        # Fine-tune with class weighting to improve recall
        # Calculate class weights to help with potential imbalance
        n_negative = np.sum(y_train == 0)
        n_positive = np.sum(y_train == 1)
        
        # Calculate balanced class weights with emphasis on TB class (class 1)
        # This helps improve recall for TB detection
        total = n_negative + n_positive
        weight_for_0 = (1 / n_negative) * (total / 2.0)
        weight_for_1 = (1 / n_positive) * (total / 2.0) * 1.2  # Boost TB class weight
        
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Fine-tuning with class weights: {class_weight}")
        
        # Fine-tune the model with class weights
        history2 = self.server_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs // 2 + epochs % 2,  # Second half of epochs (plus remainder)
            batch_size=batch_size,
            class_weight=class_weight
        )
        
        # Combine histories
        combined_history = {}
        for key in history1.history:
            if key in history2.history:
                combined_history[key] = history1.history[key] + history2.history[key]
        
        # Save fine-tuned model
        self.server_model.save_model('fl_models/server_model_fine_tuned.h5')
        
        # Evaluate the fine-tuned model
        metrics = self.server_model.evaluate(X_val, y_val)
        print("\nFine-tuned model performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return combined_history 