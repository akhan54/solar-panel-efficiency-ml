# src/features/selection.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class FeatureSelector:
    """Simple, robust feature selection that handles all data types."""
    
    def __init__(self):
        self.config = get_config()
        self.selected_features = {}
        self.feature_scores = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       methods: List[str] = None) -> Dict[str, List[str]]:
        """Apply feature selection methods with robust error handling."""
        logger.info(f"Starting feature selection on {X.shape[1]} features")
        
        # Get only numerical features for selection
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Found {len(numerical_features)} numerical features for selection")
        
        if len(numerical_features) == 0:
            logger.warning("No numerical features found, returning all features")
            return {'all_features': list(X.columns)}
        
        X_num = X[numerical_features].copy()
        
        # Handle missing values in numerical features only
        imputer = SimpleImputer(strategy='median')
        X_clean = pd.DataFrame(
            imputer.fit_transform(X_num),
            columns=numerical_features,
            index=X_num.index
        )
        
        # Replace any remaining infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        results = {}
        
        # Method 1: Univariate selection (most robust)
        try:
            logger.info("Applying univariate feature selection")
            selector = SelectKBest(score_func=f_regression, k=min(25, len(numerical_features)))
            selector.fit(X_clean, y)
            selected = X_clean.columns[selector.get_support()].tolist()
            results['univariate'] = selected
            logger.info(f"Univariate selected {len(selected)} features")
        except Exception as e:
            logger.warning(f"Univariate selection failed: {e}")
            results['univariate'] = numerical_features[:25]
        
        # Method 2: Tree-based selection
        try:
            logger.info("Applying tree-based feature selection")
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X_clean, y)
            
            # Get top features by importance
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [numerical_features[i] for i in indices[:20]]
            results['tree_based'] = top_features
            logger.info(f"Tree-based selected {len(top_features)} features")
        except Exception as e:
            logger.warning(f"Tree-based selection failed: {e}")
            results['tree_based'] = numerical_features[:20]
        
        # Method 3: Correlation-based selection
        try:
            logger.info("Applying correlation-based feature selection")
            correlations = X_clean.corrwith(y).abs()
            top_corr_features = correlations.nlargest(15).index.tolist()
            results['correlation'] = top_corr_features
            logger.info(f"Correlation-based selected {len(top_corr_features)} features")
        except Exception as e:
            logger.warning(f"Correlation selection failed: {e}")
            results['correlation'] = numerical_features[:15]
        
        self.selected_features = results
        return results
    
    def get_consensus_features(self, min_votes: int = 2) -> List[str]:
        """Get features selected by multiple methods."""
        if not self.selected_features:
            logger.warning("No feature selection results available")
            return []
        
        # Count votes for each feature
        feature_votes = {}
        for method, features in self.selected_features.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features with enough votes
        consensus_features = [feat for feat, votes in feature_votes.items() 
                            if votes >= min_votes]
        
        # Ensure minimum features
        if len(consensus_features) < 10:
            # Get most voted features
            sorted_by_votes = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            consensus_features = [feat for feat, _ in sorted_by_votes[:15]]
        
        logger.info(f"Selected {len(consensus_features)} consensus features")
        return consensus_features
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """Create a simple feature ranking."""
        if not self.selected_features:
            return pd.DataFrame()
        
        all_features = set()
        for features in self.selected_features.values():
            all_features.update(features)
        
        ranking_data = []
        for feature in all_features:
            votes = sum(1 for features in self.selected_features.values() if feature in features)
            ranking_data.append({'feature': feature, 'vote_count': votes})
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('vote_count', ascending=False)
        return ranking_df