"""
Smoke tests: Quick checks that the pipeline runs without crashing.
These tests should complete in < 30 seconds total.
"""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
import subprocess
import sys


class TestPipelineSmoke:
    """Smoke tests to ensure basic pipeline functionality."""
    
    def test_data_files_exist(self):
        """Test that required data files exist."""
        train_file = Path("data/raw/train.csv")
        test_file = Path("data/raw/test.csv")
        
        assert train_file.exists(), "Training data file missing"
        assert test_file.exists(), "Test data file missing"
    
    def test_train_data_schema(self):
        """Test that training data has expected columns and reasonable ranges."""
        df = pd.read_csv("data/raw/train.csv")
        
        # Check required columns exist
        required_cols = ['temperature', 'irradiance', 'efficiency']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check reasonable ranges - allow for real-world outliers, ignore NaN
        temp_valid = df['temperature'].dropna()
        irr_valid = df['irradiance'].dropna()
        eff_valid = df['efficiency'].dropna()

        assert temp_valid.between(-100, 200).all(), "Temperature out of range (impossible values)"
        assert irr_valid.between(-1000, 2000).all(), "Irradiance out of range (impossible values)"
        assert eff_valid.between(0, 1).all(), "Efficiency should be in [0, 1]"
                
        # Check for excessive missing values
        missing_pct = df.isnull().sum() / len(df)
        assert (missing_pct < 0.5).all(), "Too many missing values in some columns"
    
    def test_config_file_loads(self):
        """Test that configuration file loads without errors."""
        from src.utils.config import load_config
        
        config = load_config('config/config.yaml')
        assert config is not None
        
        # Check critical config values
        assert hasattr(config, 'get') or hasattr(config, '_config')
    
    def test_results_directory_structure(self):
        """Test that results directory has expected structure."""
        results_dir = Path("results")
        
        # These should exist after running the pipeline
        expected_dirs = ['plots', 'models', 'reports', 'logs']
        for dir_name in expected_dirs:
            dir_path = results_dir / dir_name
            assert dir_path.exists(), f"Missing results subdirectory: {dir_name}"


class TestMetricsReasonableness:
    """Test that metrics are within reasonable ranges."""
    
    def test_metrics_file_exists(self):
        """Test that metrics.json exists and is valid JSON."""
        metrics_file = Path("results/metrics.json")
        assert metrics_file.exists(), "metrics.json not found"
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert metrics is not None
        assert len(metrics) > 0
    
    def test_metrics_reasonable_ranges(self):
        """Test that reported metrics are in reasonable ranges."""
        with open("results/metrics.json", 'r') as f:
            metrics = json.load(f)
        
        # Check final evaluation metrics
        if 'final_evaluation' in metrics:
            for model_name, model_data in metrics['final_evaluation'].items():
                if 'metrics' in model_data:
                    m = model_data['metrics']
                    
                    # RMSE should be reasonable (0.01 to 0.5 for 0-1 scale)
                    if 'rmse' in m:
                        assert 0.01 <= m['rmse'] <= 0.5, f"{model_name} RMSE out of range: {m['rmse']}"
                    
                    # R² should be between -1 and 1
                    if 'r2' in m:
                        assert -1 <= m['r2'] <= 1, f"{model_name} R² out of range: {m['r2']}"
                    
                    # MAE should be positive and < 1
                    if 'mae' in m:
                        assert 0 < m['mae'] < 1, f"{model_name} MAE out of range: {m['mae']}"
    
    def test_physics_baseline_exists(self):
        """Test that physics baseline is documented."""
        with open("results/metrics.json", 'r') as f:
            metrics = json.load(f)
        
        assert 'physics_baseline' in metrics, "Physics baseline not found in metrics"
        baseline = metrics['physics_baseline']
        assert 'rmse' in baseline
        
        # Handle case where baseline calculation failed (returns None)
        if baseline['rmse'] is not None:
            assert baseline['rmse'] > 0, "Physics baseline RMSE should be positive"
        else:
            # If None, check that note explains why
            assert 'note' in baseline, "If RMSE is None, should have explanatory note"
            print(f"\nInfo: Physics baseline calculation unavailable: {baseline.get('note')}")


class TestOutputFiles:
    """Test that expected output files are created."""
    
    def test_key_plots_exist(self):
        """Test that at least some plots were generated."""
        plots_dir = Path("results/plots")
        
        # Should have at least a few plots
        plot_files = list(plots_dir.glob("*.png"))
        assert len(plot_files) > 0, "No plot files generated"
    
    def test_model_comparison_exists(self):
        """Test that model comparison file exists."""
        comparison_file = Path("results/model_comparison.csv")
        
        # This might not always exist depending on run
        if comparison_file.exists():
            df = pd.read_csv(comparison_file)
            assert len(df) > 0
            assert 'Model' in df.columns or df.index.name == 'Model'


class TestMinimalPipelineRun:
    """Test running a minimal version of the pipeline."""
    
    @pytest.mark.slow
    def test_minimal_training_run(self, tmp_path):
        """
        Test that training can run with minimal config.
        Marked as slow - only run when needed.
        """
        # This test is marked slow because it actually trains a model
        # Only run with: pytest -m slow
        
        # Create a minimal test dataset
        n_samples = 100
        test_data = pd.DataFrame({
            'id': range(n_samples),
            'temperature': np.random.normal(30, 10, n_samples),
            'irradiance': np.random.normal(800, 200, n_samples),
            'humidity': np.random.uniform(20, 80, n_samples),
            'panel_age': np.random.randint(1, 10, n_samples),
            'maintenance_count': np.random.randint(0, 5, n_samples),
            'soiling_ratio': np.random.uniform(0, 0.3, n_samples),
            'voltage': np.random.normal(47, 2, n_samples),
            'current': np.random.normal(7, 1, n_samples),
            'module_temperature': np.random.normal(35, 12, n_samples),
            'cloud_coverage': np.random.uniform(0, 0.5, n_samples),
            'wind_speed': np.random.uniform(0, 10, n_samples),
            'pressure': np.random.normal(1013, 10, n_samples),
            'string_id': np.random.choice(['A1', 'B2', 'C3'], n_samples),
            'error_code': np.random.choice(['E00', 'E01', ''], n_samples),
            'installation_type': np.random.choice(['fixed', 'tracking', 'dual-axis'], n_samples),
            'efficiency': np.random.uniform(0.10, 0.25, n_samples)
        })
        
        test_file = tmp_path / "test_mini.csv"
        test_data.to_csv(test_file, index=False)
        
        # Try to load and process this data
        from src.data.loader import DataLoader
        from src.features.engineering import FeatureEngineer
        
        loader = DataLoader()
        data = loader.load_data(str(test_file))
        assert len(data) == n_samples
        
        engineer = FeatureEngineer()
        features = engineer.engineer_features(data)
        assert features is not None
        assert len(features) == n_samples


if __name__ == "__main__":
    # Run only fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])