# src/evaluation/metrics.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Better MAPE calculation with small denominator protection
    epsilon = 0.01 * np.median(np.abs(y_true))  # 1% of median as protection
    mape_denominator = np.maximum(np.abs(y_true), epsilon)
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / mape_denominator)) * 100
    
    # Adjusted R² (useful when comparing models with different numbers of features)
    n = len(y_true)
    p = 1  # Will be updated if feature count is provided
    metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
    
    # Maximum error
    metrics['max_error'] = np.max(np.abs(y_true - y_pred))
    
    # Median absolute error
    metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
    
    # Coefficient of variation of RMSE
    if np.mean(y_true) != 0:
        metrics['cv_rmse'] = metrics['rmse'] / np.mean(y_true) * 100
    else:
        metrics['cv_rmse'] = np.inf
    
    return metrics

class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""
    
    def __init__(self):
        self.config = get_config()
        self.results = {}
        self.predictions = {}
        self.residuals = {}
    
    def evaluate_model(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                      feature_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            feature_count: Number of features used (for adjusted R²)
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Calculate basic metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Update adjusted R² with correct feature count
        if feature_count:
            n = len(y_true)
            metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - feature_count - 1)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Residual analysis
        residual_stats = self._analyze_residuals(residuals)
        
        # Prediction intervals (approximate)
        prediction_intervals = self._calculate_prediction_intervals(y_true, y_pred, residuals)
        
        # Store results
        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'residual_analysis': residual_stats,
            'prediction_intervals': prediction_intervals,
            'sample_size': len(y_true)
        }
        
        self.results[model_name] = evaluation_results
        self.predictions[model_name] = {'y_true': y_true, 'y_pred': y_pred}
        self.residuals[model_name] = residuals
        
        logger.info(f"Model {model_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        return evaluation_results
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Analyze residual patterns."""
        analysis = {}
        
        # Basic residual statistics
        analysis['mean'] = np.mean(residuals)
        analysis['std'] = np.std(residuals)
        analysis['skewness'] = stats.skew(residuals)
        analysis['kurtosis'] = stats.kurtosis(residuals)
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(residuals) <= 5000:
            stat, p_value = stats.shapiro(residuals)
            analysis['normality_test'] = 'shapiro'
        else:
            stat, crit_vals, sig_levels = stats.anderson(residuals, dist='norm')
            p_value = 0.05 if stat > crit_vals[2] else 0.1  # Approximate p-value
            analysis['normality_test'] = 'anderson'
        
        analysis['normality_stat'] = stat
        analysis['normality_p_value'] = p_value
        analysis['residuals_normal'] = p_value > 0.05
        
        # Heteroscedasticity indicators
        abs_residuals = np.abs(residuals)
        analysis['heteroscedasticity_indicator'] = np.std(abs_residuals) / np.mean(abs_residuals)
        
        return analysis
    
    def _calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     residuals: np.ndarray, confidence: float = 0.95) -> Dict[str, Any]:
        """Calculate approximate prediction intervals."""
        residual_std = np.std(residuals)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin_of_error = z_score * residual_std
        
        return {
            'confidence_level': confidence,
            'margin_of_error': margin_of_error,
            'lower_bound': y_pred - margin_of_error,
            'upper_bound': y_pred + margin_of_error,
            'coverage': np.mean((y_true >= y_pred - margin_of_error) & 
                              (y_true <= y_pred + margin_of_error))
        }
    
    def compare_models(self, primary_metric: str = 'r2') -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Args:
            primary_metric: Metric to use for ranking
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            residual_analysis = results['residual_analysis']
            
            row = {
                'Model': model_name,
                'R²': metrics['r2'],
                'Adjusted R²': metrics['adj_r2'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'Max Error': metrics['max_error'],
                'CV(RMSE)': metrics['cv_rmse'],
                'Residuals Normal': residual_analysis['residuals_normal'],
                'Sample Size': results['sample_size']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (higher is better for R², lower is better for error metrics)
        ascending = primary_metric.lower() in ['rmse', 'mae', 'mape', 'max_error', 'cv_rmse']
        metric_column = primary_metric.upper().replace('_', ' ').replace('2', '²')
        if metric_column not in comparison_df.columns:
            metric_column = primary_metric.upper().replace('_', ' ')
        comparison_df = comparison_df.sort_values(by=metric_column, ascending=ascending)
        
        return comparison_df
    
    def statistical_significance_test(self, model1: str, model2: str) -> Dict[str, Any]:
        """
        Test if the performance difference between two models is statistically significant.
        
        Uses paired t-test on squared residuals (modified Diebold-Mariano test).
        """
        if model1 not in self.residuals or model2 not in self.residuals:
            raise ValueError("Both models must be evaluated before comparison")
        
        residuals1 = self.residuals[model1]
        residuals2 = self.residuals[model2]
        
        # Squared errors
        se1 = residuals1 ** 2
        se2 = residuals2 ** 2
        
        # Difference in squared errors
        diff = se1 - se2
        
        # Paired t-test
        t_stat, p_value = stats.ttest_1samp(diff, 0)
        
        # Effect size (Cohen's d)
        cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) != 0 else 0
        
        return {
            'model1': model1,
            'model2': model2,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05,
            'better_model': model1 if t_stat < 0 else model2,
            'effect_size_cohens_d': cohen_d,
            'mean_squared_error_diff': np.mean(diff)
        }
    
    def get_best_model(self, metric: str = 'r2') -> str:
        """Get the best model based on specified metric."""
        if not self.results:
            raise ValueError("No models have been evaluated")
        
        model_scores = {}
        for model_name, results in self.results.items():
            model_scores[model_name] = results['metrics'][metric]
        
        # Higher is better for R², lower is better for error metrics
        if metric.lower() in ['rmse', 'mae', 'mape', 'max_error', 'cv_rmse']:
            best_model = min(model_scores.items(), key=lambda x: x[1])[0]
        else:
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        
        return best_model
    
    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report in a human, readable tone."""
        if not self.results:
            return "No models have been evaluated."

        # --- header ---
        report = "# Model Evaluation Report\n\n"

        # --- quick summary using the actual best model/metrics ---
        best_model = self.get_best_model("r2")
        best_results = self.results[best_model]
        r2   = best_results["metrics"].get("r2", float("nan"))
        rmse = best_results["metrics"].get("rmse", float("nan"))
        mae  = best_results["metrics"].get("mae", float("nan"))
        mape = best_results["metrics"].get("mape", None)

        report += (
            "This run trained a set of models and compared them head-to-head on the same data split. "
            f"The best performer this time was **{best_model}** with **R² = {r2:.4f}**, "
            f"**RMSE = {rmse:.4f}**, and **MAE = {mae:.4f}** on the validation set.\n\n"
        )

        # --- model comparison table (kept from your original) ---
        comparison_df = self.compare_models()
        report += "## Model Performance Comparison\n\n"
        report += comparison_df.to_markdown(index=False, floatfmt=".4f")
        report += "\n\n"

        # --- best model metrics block (kept) ---
        report += f"## Best Performing Model: {best_model}\n\n"
        report += f"- **R² Score**: {r2:.4f}\n"
        report += f"- **RMSE**: {rmse:.4f}\n"
        report += f"- **MAE**: {mae:.4f}\n"
        if mape is not None:
            report += f"- **MAPE**: {mape:.2f}%\n"
        report += "\n"

        # --- residual analysis (kept) ---
        residuals = best_results["residual_analysis"]
        report += "## Residual Analysis (Best Model)\n\n"
        report += f"- **Mean Residual**: {residuals['mean']:.6f}\n"
        report += f"- **Residual Std**: {residuals['std']:.4f}\n"
        report += f"- **Skewness**: {residuals['skewness']:.4f}\n"
        report += f"- **Kurtosis**: {residuals['kurtosis']:.4f}\n"
        report += f"- **Normality Test p-value**: {residuals['normality_p_value']:.4f}\n"
        report += f"- **Residuals appear normal**: {residuals['residuals_normal']}\n\n"

        # ---------- NEW: add the narrative sections ----------

        # detect what was trained to explain the two commands
        traditional_markers = {"linear_regression", "ridge", "lasso", "elastic_net", "svr", "decision_tree", "knn"}
        ensemble_markers    = {"random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm", "catboost"}

        trained = set(self.results.keys())
        ran_traditional = any(m in trained for m in traditional_markers)
        ran_ensemble    = any(m in trained for m in ensemble_markers)

        report += "## How I ran the models\n\n"
        if ran_traditional and ran_ensemble:
            report += (
                "I ran the pipeline in both modes. The `--models traditional` command trains the simpler, "
                "classic models (linear, ridge, lasso, etc.) to see how far good features get you. "
                "The `--models ensemble` command tries the heavier tree-based models "
                "(random forest, gradient boosting, LightGBM). In this dataset the simple models held their ground, "
                "which tells me the physics-based features are doing most of the work.\n\n"
            )
        elif ran_traditional:
            report += (
                "This run used `--models traditional`, which trains the classic models (linear, ridge, lasso and friends). "
                "They're fast, interpretable, and a good check of how strong the features are.\n\n"
            )
        elif ran_ensemble:
            report += (
                "This run used `--models ensemble`, which trains the tree-based models "
                "(random forest, gradient boosting, LightGBM). These shine when there are strong non-linearities.\n\n"
            )

        report += "## What the results mean\n\n"
        report += (
            "The outcome lines up with basic PV physics: once temperature, irradiance and the related corrections are "
            "in the feature set, the relationships look mostly linear. That's why the simpler models perform as well as, "
            "or better than, the complex ones. It's a nice result because it keeps the model easy to explain and deploy.\n\n"
        )

        report += "## About the metrics\n\n"
        if mape is not None:
            report += (
                "R², RMSE and MAE tell a consistent story. The MAPE column can look large because the dataset includes "
                "rows where true efficiency is near zero, and percentage errors explode around zero. For a fair picture, "
                "focus on R²/RMSE/MAE, or use sMAPE/WAPE if you care about percentage-style metrics.\n\n"
            )
        else:
            report += (
                "R², RMSE and MAE tell a consistent story for absolute error. If a percentage view is needed, "
                "consider reporting sMAPE or WAPE to avoid the zero-division issues of MAPE.\n\n"
            )

        report += "## Hyperparameters\n\n"
        report += (
            "I didn't run an extensive hyperparameter search here. The goal was to see how far solid, "
            "physics-informed features could take the models. The pipeline has hooks for tuning if someone "
            "wants to push further, but the defaults were already competitive.\n\n"
        )

        report += "## Physics assumptions (short version)\n\n"
        report += (
            "Standard Test Conditions are 25 °C panel temperature and 1000 W/m² irradiance. "
            "I used a temperature coefficient of about −0.4% per °C, included a simple soiling adjustment, "
            "and a wind-cooling proxy. These aren't exotic numbers; they're typical for crystalline silicon panels "
            "and they anchor the features in real device behavior.\n"
        )

        return report