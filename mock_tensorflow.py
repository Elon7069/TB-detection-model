"""
Mock TensorFlow module for demonstration purposes.
This allows the TB Detection project to run for demonstration without requiring actual TensorFlow.
"""

import numpy as np
import os

print("Using mock TensorFlow module for demonstration purposes.")
print("This is a limited implementation that allows the main code to run.")
print("For actual model training and inference, please install TensorFlow properly.")

# Mock keras classes and functions
class Sequential:
    def __init__(self, layers=None):
        self.layers = layers or []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, **kwargs):
        pass
        
    def fit(self, x, y, **kwargs):
        class History:
            def __init__(self):
                self.history = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4]}
        return History()
        
    def predict(self, x):
        # Return random predictions
        return np.random.random((len(x), 1))
        
    def evaluate(self, x, y):
        return [0.2, 0.8]  # loss, accuracy
        
    def save(self, filepath):
        # Create an empty file
        with open(filepath, 'w') as f:
            f.write("Mock model")
        
    def load_weights(self, filepath):
        pass
        
    def get_weights(self):
        return [np.random.random((10, 10)) for _ in range(5)]
        
    def set_weights(self, weights):
        pass

# Mock keras layers
class Dense:
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = activation
        
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        
class GlobalAveragePooling2D:
    def __init__(self, **kwargs):
        pass

# Mock applications
class applications:
    class ResNet50:
        def __init__(self, weights=None, include_top=True, input_shape=None):
            self.weights = weights
            self.include_top = include_top
            self.input_shape = input_shape
            self.layers = [Mock() for _ in range(10)]
            self.output = Mock()
            self.input = Mock()
            
    class preprocess_input:
        @staticmethod
        def __call__(x):
            return x

# Mock keras Model
class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.metrics_names = ['loss', 'accuracy', 'precision', 'recall']
        
    def compile(self, **kwargs):
        pass
        
    def fit(self, x, y=None, **kwargs):
        class History:
            def __init__(self):
                self.history = {'loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4]}
        return History()
        
    def predict(self, x):
        # Return random predictions between 0 and 1
        if isinstance(x, list):
            return np.random.random((len(x[0]), 1))
        return np.random.random((len(x), 1))
        
    def evaluate(self, x, y):
        return [0.2, 0.8, 0.75, 0.83]  # loss, accuracy, precision, recall
        
    def save(self, filepath):
        # Create an empty file
        with open(filepath, 'w') as f:
            f.write("Mock model")
            
    def save_weights(self, filepath):
        # Create an empty file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write("Mock weights")
            
    def load_weights(self, filepath):
        pass
        
    def get_weights(self):
        return [np.random.random((10, 10)) for _ in range(5)]
        
    def set_weights(self, weights):
        pass
        
    def get_layer(self, name):
        return Mock()

# Mock optimizers
class optimizers:
    class Adam:
        def __init__(self, learning_rate=0.001):
            self.learning_rate = learning_rate

# Mock callback classes
class callbacks:
    class ModelCheckpoint:
        def __init__(self, filepath, **kwargs):
            self.filepath = filepath
            
    class EarlyStopping:
        def __init__(self, **kwargs):
            pass
            
    class ReduceLROnPlateau:
        def __init__(self, **kwargs):
            pass

# Mock metrics
class metrics:
    class Precision:
        def __init__(self):
            pass
            
    class Recall:
        def __init__(self):
            pass

# Mock class for anything else needed
class Mock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __call__(self, *args, **kwargs):
        return Mock()
        
    def __getattr__(self, name):
        return Mock()

# Create mock keras module
keras = type('keras', (), {
    'Sequential': Sequential,
    'Model': Model,
    'layers': type('layers', (), {
        'Dense': Dense,
        'Dropout': Dropout,
        'GlobalAveragePooling2D': GlobalAveragePooling2D,
    }),
    'applications': applications,
    'optimizers': optimizers,
    'callbacks': callbacks,
    'metrics': metrics,
    'preprocessing': type('preprocessing', (), {
        'image': type('image', (), {
            'ImageDataGenerator': lambda **kwargs: Mock(),
        }),
    }),
})

# Mock tensorflow.GradientTape
class GradientTape:
    def __init__(self):
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def watch(self, tensor):
        pass
        
    def gradient(self, target, sources):
        # Return a random gradient of the same shape as the source
        if isinstance(sources, list):
            return [np.random.random(source.shape) for source in sources]
        return np.random.random(sources.shape)

# Provide required globals
random = np.random

# Create the main mock TensorFlow module
__version__ = "2.10.0-mock" 