import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import os

class TBModel:
    """
    TB detection model using ResNet50 as the backbone.
    """
    
    def __init__(self, input_shape=(224, 224, 3), weights_path=None):
        """
        Initialize the TB detection model.
        
        Args:
            input_shape: Input shape of the images (height, width, channels)
            weights_path: Path to pre-trained weights file (optional)
        """
        self.input_shape = input_shape
        self.weights_path = weights_path
        self.model = self._build_model()
        
        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
    
    def _build_model(self):
        """
        Build the model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet50 model
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        
        # Add spatial attention mechanism to focus on relevant regions
        # This helps improve recall by focusing on all potential TB indicators
        spatial_attention = GlobalAveragePooling2D()(x)
        spatial_attention = Dense(256, activation='relu')(spatial_attention)
        spatial_attention = Dense(x.shape[-1], activation='sigmoid')(spatial_attention)
        spatial_attention = tf.reshape(spatial_attention, [-1, 1, 1, x.shape[-1]])
        x = x * spatial_attention  # Apply attention
        
        # Global pooling
        x = GlobalAveragePooling2D()(x)
        
        # First dense block with high capacity
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Second dense block with skip connection (ResNet-style)
        # Skip connections help maintain gradient flow, improving learning
        skip = x
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = tf.keras.layers.add([x, skip])  # Add skip connection
        
        # Final dense block
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create the final model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model with recall-focused metrics
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                Precision(),
                Recall(),
                tf.keras.metrics.AUC(),  # AUC helps evaluate threshold-independent performance
                self._f1_score  # Custom F1 score metric
            ]
        )
        
        return model
    
    def _f1_score(self, y_true, y_pred):
        """
        Custom F1 score metric implementation for TensorFlow.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            F1 score
        """
        # Calculate precision and recall
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()
        
        precision.update_state(y_true, y_pred)
        recall.update_state(y_true, y_pred)
        
        # Calculate F1 score
        p = precision.result()
        r = recall.result()
        
        # Handle division by zero
        f1 = tf.math.divide_no_nan(2 * p * r, p + r)
        
        return f1
    
    def unfreeze_layers(self, num_layers=10):
        """
        Unfreeze the last n layers of the base model for fine-tuning.
        
        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        # Find the ResNet50 layers
        res_layers = [layer for layer in self.model.layers if 'res' in layer.name.lower()]
        
        # Compute the number of layers to unfreeze
        num_to_unfreeze = min(num_layers, len(res_layers))
        
        # Unfreeze the last n layers
        for layer in res_layers[-num_to_unfreeze:]:
            layer.trainable = True
        
        # Recompile the model with a lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        print(f"Unfroze the last {num_to_unfreeze} ResNet layers for fine-tuning")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, callbacks=None, class_weight=None):
        """
        Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: Custom callbacks (optional)
            class_weight: Class weights for imbalanced data (optional)
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Calculate class weights if not provided (helps improve recall for minority class)
        if class_weight is None:
            n_negative = np.sum(y_train == 0)
            n_positive = np.sum(y_train == 1)
            
            # Only apply class weighting if there's an imbalance
            if n_negative != n_positive:
                # Calculate balanced class weights
                total = n_negative + n_positive
                weight_for_0 = (1 / n_negative) * (total / 2.0)
                weight_for_1 = (1 / n_positive) * (total / 2.0)
                
                # Apply higher weight to TB class to improve recall
                weight_for_1 *= 1.2  # Slight boost to TB class weight
                
                class_weight = {0: weight_for_0, 1: weight_for_1}
                
                print(f"Applying class weights: {class_weight}")
        
        # Create data augmentation generator for training
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Train the model using the augmented data
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return history
    
    def _get_default_callbacks(self):
        """
        Get default training callbacks.
        
        Returns:
            List of default callbacks
        """
        # Create model checkpoint callback
        checkpoint = ModelCheckpoint(
            'tb_model_best.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        # Create early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        return [checkpoint, early_stopping, reduce_lr]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Evaluate the model
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        # Calculate additional metrics
        y_pred = self.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate F1-score
        precision = metrics['precision']
        recall = metrics['recall']
        if precision + recall > 0:
            metrics['f1'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1'] = 0.0
        
        return metrics
    
    def predict(self, X):
        """
        Generate predictions.
        
        Args:
            X: Input images
            
        Returns:
            Probability predictions
        """
        return self.model.predict(X, verbose=0)
    
    def save_weights(self, filepath):
        """
        Save model weights.
        
        Args:
            filepath: Path to save the weights
        """
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")
    
    def save_model(self, filepath):
        """
        Save the entire model.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_model_for_gradcam(self):
        """
        Get the model for GradCAM visualization.
        
        Returns:
            The Keras model
        """
        return self.model 