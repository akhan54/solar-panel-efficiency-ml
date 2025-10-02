import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class DataValidator:
    """Validates data quality and performs checks."""
    
    def __init__(self):
        self.config = get_config()
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting validation for {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'basic_checks': self._basic_checks(df),
            'missing_data': self._check_missing_data(df),
            'data_types': self._check_data_types(df),
            'outliers': self._detect_outliers(df),
            'duplicates': self._check_duplicates(df),
            'data_quality_score': 0
        }
        
        # Calculate overall data quality score
        results['data_quality_score'] = self._calculate_quality_score(results)
        
        self.validation_results[dataset_name] = results
        logger.info(f"Validation completed for {dataset_name}. Quality score: {results['data_quality_score']:.2f}")
        
        return results
    
    def _basic_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data structure checks."""
        return {
            'shape': df.shape,
            'empty_dataset': len(df) == 0,
            'all_columns_empty': df.isnull().all().any(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentages = missing_counts / len(df) * 100
        
        threshold = self.config.get('preprocessing.missing_threshold', 0.3) * 100
        
        return {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict(),
            'high_missing_columns': missing_percentages[missing_percentages > threshold].index.tolist(),
            'missing_pattern_exists': self._check_missing_patterns(df)
        }
    
    def _check_missing_patterns(self, df: pd.DataFrame) -> bool:
        """Check if there are systematic missing data patterns."""
        # Simple check: are there rows with multiple missing values?
        missing_per_row = df.isnull().sum(axis=1)
        return (missing_per_row > 1).sum() > len(df) * 0.05  # More than 5% of rows have multiple missing
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data types against expected schema."""
        expected_numerical = self.config.get('features.numerical', [])
        expected_categorical = self.config.get('features.categorical', [])
        
        actual_types = df.dtypes.to_dict()
        
        type_issues = {}
        for col in expected_numerical:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                type_issues[col] = f"Expected numeric, got {actual_types[col]}"
        
        return {
            'actual_types': actual_types,
            'type_mismatches': type_issues,
            'unexpected_columns': [col for col in df.columns 
                                 if col not in expected_numerical + expected_categorical + 
                                 [self.config.get('data.id_column'), self.config.get('data.target_column')]]
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical columns using IQR and Z-score methods."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        threshold = self.config.get('preprocessing.outlier_threshold', 3)
        
        for col in numerical_cols:
            if col == self.config.get('data.id_column'):
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)).sum()
            
            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            z_outliers = (z_scores > threshold).sum()
            
            outlier_results[col] = {
                'iqr_outliers': int(iqr_outliers),
                'z_score_outliers': int(z_outliers),
                'outlier_percentage': float(max(iqr_outliers, z_outliers) / len(series) * 100)
            }
        
        return outlier_results
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate rows and potential data leakage."""
        total_duplicates = df.duplicated().sum()
        
        # Check duplicates excluding ID column
        id_col = self.config.get('data.id_column')
        if id_col in df.columns:
            df_no_id = df.drop(columns=[id_col])
            content_duplicates = df_no_id.duplicated().sum()
        else:
            content_duplicates = total_duplicates
        
        return {
            'total_duplicates': int(total_duplicates),
            'content_duplicates': int(content_duplicates),
            'duplicate_percentage': float(total_duplicates / len(df) * 100)
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate an overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize for missing data
        missing_data = results['missing_data']
        if missing_data['total_missing'] > 0:
            missing_penalty = min(missing_data['total_missing'] / (len(results) * 10), 20)
            score -= missing_penalty
        
        # Penalize for duplicates
        duplicate_penalty = min(results['duplicates']['duplicate_percentage'], 10)
        score -= duplicate_penalty
        
        # Penalize for type mismatches
        type_penalty = len(results['data_types']['type_mismatches']) * 5
        score -= type_penalty
        
        # Penalize for excessive outliers
        outlier_penalty = 0
        for col_outliers in results['outliers'].values():
            if col_outliers['outlier_percentage'] > 5:
                outlier_penalty += min(col_outliers['outlier_percentage'] / 2, 5)
        score -= outlier_penalty
        
        return max(score, 0.0)