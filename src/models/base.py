import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_config()
        self.model = None
        self.pipeline = None
        self.is_fitted = False
        self.feature_names = None
        self.cv_scores = None
    
    @abstractmethod
    def _create_model(self, **kwargs):
        """Create the underlying model instance."""
        pass
    
    def build_pipeline(self, preprocessor=None) -> Pipeline:
        """Build sklearn pipeline with preprocessing and model."""
        steps = []
        
        if preprocessor is not None:
            steps.append(('preprocessor', preprocessor))
        
        steps.append(('model', self.model))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        """Fit the model."""
        logger.info(f"Training {self.model_name}")
        
        if self.pipeline is not None:
            self.pipeline.fit(X, y, **fit_params)
        else:
            self.model.fit(X, y, **fit_params)
        
        self.is_fitted = True
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        logger.info(f"{self.model_name} training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before prediction")
        
        if self.pipeline is not None:
            return self.pipeline.predict(X)
        else:
            return self.model.predict(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """Perform cross-validation."""
        logger.info(f"Cross-validating {self.model_name}")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        if self.pipeline is not None:
            estimator = self.pipeline
        else:
            estimator = self.model
        
        scores = cross_val_score(estimator, X, y, cv=kfold, scoring=scoring)
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        self.cv_scores = cv_results
        logger.info(f"{self.model_name} CV {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
        
        return cv_results
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        # Try different methods to get feature importance
        model_to_check = self.pipeline.named_steps['model'] if self.pipeline else self.model
        
        importance = None
        if hasattr(model_to_check, 'feature_importances_'):
            importance = model_to_check.feature_importances_
        elif hasattr(model_to_check, 'coef_'):
            importance = np.abs(model_to_check.coef_)
        
        if importance is not None:
            return dict(zip(self.feature_names, importance))
        
        return None