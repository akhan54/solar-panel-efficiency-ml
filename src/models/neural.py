import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NeuralNetworkModel(BaseModel):
    """Multi-layer Perceptron Neural Network model."""
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100,), 
                 activation: str = 'relu', alpha: float = 0.0001,
                 learning_rate: str = 'constant', max_iter: int = 500,
                 random_state: int = 42, **kwargs):
        super().__init__("Neural Network")
        self.model = self._create_model(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, alpha=alpha,
            learning_rate=learning_rate, max_iter=max_iter,
            random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return MLPRegressor(**kwargs)

class DeepNeuralNetworkModel(BaseModel):
    """Deep Neural Network with multiple hidden layers."""
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (150, 100, 50), 
                 activation: str = 'relu', alpha: float = 0.001,
                 learning_rate: str = 'adaptive', max_iter: int = 1000,
                 random_state: int = 42, **kwargs):
        super().__init__("Deep Neural Network")
        self.model = self._create_model(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, alpha=alpha,
            learning_rate=learning_rate, max_iter=max_iter,
            random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return MLPRegressor(**kwargs)

class NeuralModels:
    """Factory class for neural network models."""
    
    @staticmethod
    def get_all_models() -> Dict[str, BaseModel]:
        """Get all neural network models with default parameters."""
        return {
            'neural_network': NeuralNetworkModel(
                hidden_layer_sizes=(100,), 
                random_state=42
            ),
            'deep_neural_network': DeepNeuralNetworkModel(
                hidden_layer_sizes=(150, 100, 50),
                random_state=42
            )
        }