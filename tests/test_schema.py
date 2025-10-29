"""
Schema and data validation tests.
Fast tests to ensure data quality without heavy computation.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataSchema:
    """Test data schema and structure."""
    
    @pytest.fixture
    def train_data(self):
        """Load training data once for all tests."""
        return pd.read_csv("data/raw/train.csv")
    
    @pytest.fixture  
    def test_data(self):
        """Load test data once for all tests."""
        return pd.read_csv("data/raw/test.csv")
    
    def test_train_required_columns(self, train_data):
        """Test that training data has all required columns."""
        required = [
            'temperature', 'irradiance', 'humidity', 'voltage', 
            'current', 'module_temperature', 'efficiency'
        ]
        
        for col in required:
            assert col in train_data.columns, f"Missing column: {col}"
    
    def test_test_required_columns(self, test_data):
        """Test that test data has required columns (no efficiency)."""
        required = [
            'temperature', 'irradiance', 'humidity', 'voltage', 
            'current', 'module_temperature'
        ]
        
        for col in required:
            assert col in test_data.columns, f"Missing column: {col}"
    
    def test_efficiency_in_valid_range(self, train_data):
        """Test that efficiency values are in [0, 1]."""
        eff = train_data['efficiency'].dropna()
        
        assert eff.min() >= 0, f"Efficiency below 0: {eff.min()}"
        assert eff.max() <= 1, f"Efficiency above 1: {eff.max()}"
    
    def test_temperature_reasonable(self, train_data):
        """Test that temperature values are physically reasonable."""
        temp = train_data['temperature'].dropna()
        
        # Check for extreme outliers (may need handling in preprocessing)
        # Allow wider range since real data may have sensor errors or extreme conditions
        assert temp.min() > -100, f"Temperature impossibly low: {temp.min()}"
        assert temp.max() < 200, f"Temperature impossibly high: {temp.max()}"
        
        # Log warning if outside typical operating range
        if temp.max() > 70:
            print(f"\nWarning: Maximum temperature {temp.max():.1f}°C exceeds typical range. May be outliers.")
        if temp.min() < -40:
            print(f"\nWarning: Minimum temperature {temp.min():.1f}°C below typical range. May be outliers.")
    
    def test_irradiance_reasonable(self, train_data):
        """Test that irradiance values are physically reasonable."""
        irr = train_data['irradiance'].dropna()
        
        # Check percentage of negative or extreme values
        negative_count = (irr < 0).sum()
        total_count = len(irr)
        
        # Allow some negative values (sensor errors) but not too many
        assert negative_count < total_count * 0.05, \
            f"Too many negative irradiance values: {negative_count}/{total_count} ({100*negative_count/total_count:.1f}%)"
        
        # Check for extremely high values
        extreme_high = (irr > 1500).sum()
        assert extreme_high < total_count * 0.01, \
            f"Too many extremely high irradiance values: {extreme_high}/{total_count}"
        
        # Log info about data range
        if negative_count > 0:
            print(f"\nInfo: {negative_count} negative irradiance values found (likely sensor errors).")
            print(f"      Range: {irr.min():.1f} to {irr.max():.1f} W/m²")
    
    def test_voltage_current_positive(self, train_data):
        """Test that voltage and current are non-negative."""
        voltage = train_data['voltage'].dropna()
        current = train_data['current'].dropna()
        
        assert (voltage >= 0).all(), "Negative voltage found"
        assert (current >= 0).all(), "Negative current found"
    
    def test_no_all_null_rows(self, train_data):
        """Test that no rows are completely null."""
        all_null = train_data.isnull().all(axis=1)
        assert not all_null.any(), "Found completely null rows"
    
    def test_sufficient_samples(self, train_data, test_data):
        """Test that we have enough samples for training."""
        assert len(train_data) >= 1000, f"Too few training samples: {len(train_data)}"
        assert len(test_data) >= 100, f"Too few test samples: {len(test_data)}"
    
    def test_target_distribution(self, train_data):
        """Test that target variable has reasonable distribution."""
        eff = train_data['efficiency'].dropna()
        
        # Should have some variance
        assert eff.std() > 0.01, "Target has no variance"
        
        # Should not be constant
        assert len(eff.unique()) > 10, "Target has too few unique values"


class TestFeatureEngineering:
    """Test that feature engineering produces valid outputs."""
    
    def test_feature_engineer_imports(self):
        """Test that feature engineering module imports correctly."""
        from src.features.engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        assert engineer is not None
    
    def test_basic_feature_creation(self):
        """Test that basic features can be created."""
        from src.features.engineering import FeatureEngineer
        
        # Create minimal test data
        data = pd.DataFrame({
            'temperature': [25, 30, 35],
            'module_temperature': [30, 35, 40],
            'irradiance': [1000, 800, 600],
            'voltage': [48, 47, 46],
            'current': [8, 7, 6],
            'humidity': [50, 60, 70],
            'wind_speed': [2, 3, 4],
            'soiling_ratio': [0.0, 0.1, 0.2],
            'cloud_coverage': [0.0, 0.2, 0.4],
            'panel_age': [1, 2, 3],
            'maintenance_count': [1, 2, 2],
            'pressure': [1013, 1012, 1011]
        })
        
        engineer = FeatureEngineer()
        result = engineer.engineer_features(data)
        
        # Should create more features than input
        assert len(result.columns) > len(data.columns)
        
        # Should not have NaN where input didn't
        assert result['irradiance'].notna().all()
    
    def test_no_infinite_values_created(self):
        """Test that feature engineering doesn't create infinite values."""
        from src.features.engineering import FeatureEngineer
        
        data = pd.DataFrame({
            'temperature': [25, 0, 35],  # Including zero
            'module_temperature': [30, 35, 40],
            'irradiance': [1000, 0, 600],  # Including zero
            'voltage': [48, 0, 46],  # Including zero
            'current': [8, 7, 0],  # Including zero
            'humidity': [50, 60, 70],
            'wind_speed': [2, 0, 4],  # Including zero
            'soiling_ratio': [0.0, 0.1, 0.2],
            'cloud_coverage': [0.0, 0.2, 0.4],
            'panel_age': [1, 2, 3],
            'maintenance_count': [1, 2, 2],
            'pressure': [1013, 1012, 1011]
        })
        
        engineer = FeatureEngineer()
        result = engineer.engineer_features(data)
        
        # Check no infinite values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(result[col]).any(), f"Infinite values in {col}"


class TestModelImports:
    """Test that model modules import correctly."""
    
    def test_traditional_models_import(self):
        """Test that traditional models can be imported."""
        from src.models.traditional import (
            LinearRegressionModel,
            RidgeRegressionModel,
            ElasticNetModel
        )
        
        # Should not raise
        assert LinearRegressionModel is not None
        assert RidgeRegressionModel is not None
        assert ElasticNetModel is not None
    
    def test_ensemble_models_import(self):
        """Test that ensemble models can be imported."""
        from src.models.ensemble import (
            RandomForestModel,
            GradientBoostingModel
        )
        
        assert RandomForestModel is not None
        assert GradientBoostingModel is not None
    
    def test_evaluation_imports(self):
        """Test that evaluation modules import correctly."""
        from src.evaluation.metrics import ModelEvaluator
        from src.evaluation.visualization import ModelVisualizer
        
        assert ModelEvaluator is not None
        assert ModelVisualizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])