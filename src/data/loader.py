import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class DataLoader:
    """Handles data loading and basic preprocessing."""
    
    def __init__(self):
        self.config = get_config()
    
    def load_datasets(self, train_path: Optional[str] = None, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing datasets.
        
        Args:
            train_path: Path to training data
            test_path: Path to testing data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if train_path is None:
            train_path = Path(self.config.get('data.raw_path')) / self.config.get('data.train_file')
        if test_path is None:
            test_path = Path(self.config.get('data.raw_path')) / self.config.get('data.test_file')
        
        logger.info(f"Loading training data from: {train_path}")
        logger.info(f"Loading test data from: {test_path}")
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Training data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Basic data cleaning
            train_df = self._clean_data(train_df)
            test_df = self._clean_data(test_df)
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Convert numeric columns, handling errors
        numerical_cols = self.config.get('features.numerical', [])
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle categorical columns
        categorical_cols = self.config.get('features.categorical', [])
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        logger.info("Basic data cleaning completed")
        return df_clean
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about the dataset."""
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Numerical statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            info['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_stats'] = {}
            for col in categorical_cols:
                info['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'value_counts': df[col].value_counts().head().to_dict()
                }
        
        return info