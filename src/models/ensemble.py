import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor,
    VotingRegressor, StackingRegressor
)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .base import BaseModel
from .traditional import LinearRegressionModel, RidgeRegressionModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest Regression model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 random_state: int = 42, **kwargs):
        super().__init__("Random Forest")
        self.model = self._create_model(
            n_estimators=n_estimators, max_depth=max_depth, 
            random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return RandomForestRegressor(**kwargs)

class GradientBoostingModel(BaseModel):
    """Gradient Boosting Regression model."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: int = 42, **kwargs):
        super().__init__("Gradient Boosting")
        self.model = self._create_model(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return GradientBoostingRegressor(**kwargs)

class XGBoostModel(BaseModel):
    """XGBoost Regression model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        super().__init__("XGBoost")
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed")
        
        self.model = self._create_model(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return xgb.XGBRegressor(**kwargs)

class LightGBMModel(BaseModel):
    """LightGBM Regression model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_state: int = 42, **kwargs):
        super().__init__("LightGBM")
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")
        
        self.model = self._create_model(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return lgb.LGBMRegressor(**kwargs)

class ExtraTreesModel(BaseModel):
    """Extra Trees Regression model."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 random_state: int = 42, **kwargs):
        super().__init__("Extra Trees")
        self.model = self._create_model(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return ExtraTreesRegressor(**kwargs)

class AdaBoostModel(BaseModel):
    """AdaBoost Regression model."""
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0,
                 random_state: int = 42, **kwargs):
        super().__init__("AdaBoost")
        self.model = self._create_model(
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=random_state, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return AdaBoostRegressor(**kwargs)

class VotingEnsembleModel(BaseModel):
    """Voting Ensemble of multiple models."""
    
    def __init__(self, base_models: Dict[str, BaseModel], **kwargs):
        super().__init__("Voting Ensemble")
        self.base_models = base_models
        estimators = [(name, model.model) for name, model in base_models.items()]
        self.model = self._create_model(estimators=estimators, **kwargs)
    
    def _create_model(self, **kwargs):
        return VotingRegressor(**kwargs)

class StackingEnsembleModel(BaseModel):
    """Stacking Ensemble with meta-learner."""
    
    def __init__(self, base_models: Dict[str, BaseModel], 
                 meta_learner: BaseModel = None, cv: int = 5, **kwargs):
        super().__init__("Stacking Ensemble")
        self.base_models = base_models
        
        if meta_learner is None:
            meta_learner = RidgeRegressionModel(alpha=1.0)
        
        estimators = [(name, model.model) for name, model in base_models.items()]
        self.model = self._create_model(
            estimators=estimators, final_estimator=meta_learner.model,
            cv=cv, **kwargs
        )
    
    def _create_model(self, **kwargs):
        return StackingRegressor(**kwargs)

class EnsembleModels:
    """Factory class for ensemble models."""
    
    @staticmethod
    def get_all_models() -> Dict[str, BaseModel]:
        """Get all ensemble models with default parameters."""
        models = {
            'random_forest': RandomForestModel(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingModel(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesModel(n_estimators=100, random_state=42),
            'ada_boost': AdaBoostModel(n_estimators=50, random_state=42)
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBoostModel(n_estimators=100, random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LightGBMModel(n_estimators=100, random_state=42)
        
        return models
    
    @staticmethod
    def create_voting_ensemble(base_models: Dict[str, BaseModel]) -> VotingEnsembleModel:
        """Create a voting ensemble from base models."""
        return VotingEnsembleModel(base_models)
    
    @staticmethod
    def create_stacking_ensemble(base_models: Dict[str, BaseModel], 
                               meta_learner: BaseModel = None) -> StackingEnsembleModel:
        """Create a stacking ensemble from base models."""
        return StackingEnsembleModel(base_models, meta_learner)