import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from .base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LinearRegressionModel(BaseModel):
    """Linear Regression model."""
    
    def __init__(self, **kwargs):
        super().__init__("Linear Regression")
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        return LinearRegression(**kwargs)

class RidgeRegressionModel(BaseModel):
    """Ridge Regression model with L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Ridge Regression")
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)
    
    def _create_model(self, **kwargs):
        return Ridge(**kwargs)

class LassoRegressionModel(BaseModel):
    """Lasso Regression model with L1 regularization."""
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__("Lasso Regression")
        self.alpha = alpha
        self.model = self._create_model(alpha=alpha, **kwargs)
    
    def _create_model(self, **kwargs):
        return Lasso(**kwargs)

class ElasticNetModel(BaseModel):
    """ElasticNet model with L1 and L2 regularization."""
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, **kwargs):
        super().__init__("ElasticNet")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = self._create_model(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
    
    def _create_model(self, **kwargs):
        return ElasticNet(**kwargs)

class DecisionTreeModel(BaseModel):
    """Decision Tree Regression model."""
    
    def __init__(self, max_depth: Optional[int] = None, random_state: int = 42, **kwargs):
        super().__init__("Decision Tree")
        self.model = self._create_model(max_depth=max_depth, random_state=random_state, **kwargs)
    
    def _create_model(self, **kwargs):
        return DecisionTreeRegressor(**kwargs)

class SVRModel(BaseModel):
    """Support Vector Regression model."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, **kwargs):
        super().__init__("Support Vector Regression")
        self.model = self._create_model(kernel=kernel, C=C, **kwargs)
    
    def _create_model(self, **kwargs):
        return SVR(**kwargs)

class KNNModel(BaseModel):
    """K-Nearest Neighbors Regression model."""
    
    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__("K-Nearest Neighbors")
        self.model = self._create_model(n_neighbors=n_neighbors, **kwargs)
    
    def _create_model(self, **kwargs):
        return KNeighborsRegressor(**kwargs)

class TraditionalModels:
    """Factory class for traditional ML models."""
    
    @staticmethod
    def get_all_models() -> Dict[str, BaseModel]:
        """Get all traditional models with default parameters."""
        return {
            'linear_regression': LinearRegressionModel(),
            'ridge': RidgeRegressionModel(alpha=1.0),
            'lasso': LassoRegressionModel(alpha=0.1),
            'elastic_net': ElasticNetModel(alpha=0.1, l1_ratio=0.5),
            'decision_tree': DecisionTreeModel(max_depth=10, random_state=42),
            'svr': SVRModel(kernel='rbf', C=1.0),
            'knn': KNNModel(n_neighbors=5)
        }
    
    @staticmethod
    def get_model(model_name: str, **params) -> BaseModel:
        """Get a specific model with custom parameters."""
        model_classes = {
            'linear_regression': LinearRegressionModel,
            'ridge': RidgeRegressionModel,
            'lasso': LassoRegressionModel,
            'elastic_net': ElasticNetModel,
            'decision_tree': DecisionTreeModel,
            'svr': SVRModel,
            'knn': KNNModel
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model_classes[model_name](**params)