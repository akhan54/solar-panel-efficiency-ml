import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class ModelVisualizer:
    """Create visualizations for model evaluation and analysis."""
    
    def __init__(self, save_path: str = "results/plots/"):
        self.config = get_config()
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        if PLOTLY_AVAILABLE:
            self.colors = px.colors.qualitative.Set1
        else:
            self.colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metrics: List[str] = None,
                            save_name: str = "model_comparison.html") -> Optional[Any]:
        """Create model comparison visualization."""
        if metrics is None:
            metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        
        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            logger.warning("No specified metrics found in comparison dataframe")
            return self._create_matplotlib_comparison(comparison_df, save_name)
        
        if PLOTLY_AVAILABLE:
            return self._create_plotly_comparison(comparison_df, available_metrics, save_name)
        else:
            return self._create_matplotlib_comparison(comparison_df, save_name)
    
    def _create_plotly_comparison(self, comparison_df: pd.DataFrame, 
                                metrics: List[str], save_name: str):
        """Create Plotly comparison chart."""
        try:
            # Create subplots
            n_metrics = min(len(metrics), 4)
            rows = 2 if n_metrics > 2 else 1
            cols = 2 if n_metrics > 1 else 1
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=metrics[:n_metrics]
            )
            
            for i, metric in enumerate(metrics[:n_metrics]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                # Determine if higher or lower is better
                ascending = metric.upper() in ['RMSE', 'MAE', 'MAPE', 'MAX ERROR']
                
                sorted_df = comparison_df.sort_values(by=metric, ascending=not ascending)
                
                fig.add_trace(
                    go.Bar(
                        x=sorted_df['Model'],
                        y=sorted_df[metric],
                        name=metric,
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title_text="Model Performance Comparison",
                height=700,
                showlegend=False
            )
            
            # Save plot
            fig.write_html(str(self.save_path / save_name))
            logger.info(f"Model comparison plot saved to {self.save_path / save_name}")
            
            return fig
        except Exception as e:
            logger.warning(f"Could not create Plotly comparison chart: {e}")
            return self._create_matplotlib_comparison(comparison_df, save_name)
    
    def _create_matplotlib_comparison(self, comparison_df: pd.DataFrame, save_name: str):
        """Create matplotlib comparison chart as fallback."""
        try:
            metrics = ['R²', 'RMSE', 'MAE']
            available_metrics = [m for m in metrics if m in comparison_df.columns]
            
            if not available_metrics:
                available_metrics = [col for col in comparison_df.columns if col != 'Model'][:3]
            
            n_metrics = len(available_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
            
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                # Sort data
                ascending = metric.upper() in ['RMSE', 'MAE', 'MAPE', 'MAX ERROR']
                sorted_df = comparison_df.sort_values(by=metric, ascending=not ascending)
                
                # Create bar plot
                bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
                ax.set_title(f'Model Comparison - {metric}')
                ax.set_ylabel(metric)
                ax.set_xticks(range(len(sorted_df)))
                ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
                
                # Color bars
                for j, bar in enumerate(bars):
                    bar.set_color(self.colors[j % len(self.colors)])
            
            plt.tight_layout()
            
            # Save as PNG instead of HTML
            png_name = save_name.replace('.html', '.png')
            plt.savefig(self.save_path / png_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison plot saved to {self.save_path / png_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Could not create comparison visualization: {e}")
            return None
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str, save_name: str = None) -> Optional[Any]:
        """Create predictions vs actual values plot."""
        if save_name is None:
            save_name = f"predictions_vs_actual_{model_name.replace(' ', '_').lower()}.png"
        
        try:
            # Calculate metrics for display
            from .metrics import calculate_metrics
            metrics = calculate_metrics(y_true, y_pred)
            
            # Create matplotlib plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Scatter plot
            scatter = ax.scatter(y_true, y_pred, 
                               c=np.abs(y_true - y_pred), 
                               cmap='viridis', alpha=0.6)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Labels and title
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Predictions vs Actual Values - {model_name}\n'
                        f'R² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
            ax.legend()
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Absolute Error')
            
            plt.tight_layout()
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Predictions vs actual plot saved to {self.save_path / save_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Could not create predictions vs actual plot: {e}")
            return None
    
    def plot_residuals_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str, save_name: str = None) -> Optional[Any]:
        """Create comprehensive residual analysis plots."""
        if save_name is None:
            save_name = f"residuals_analysis_{model_name.replace(' ', '_').lower()}.png"
        
        try:
            residuals = y_true - y_pred
            
            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Residuals vs Predicted
            axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Predicted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Predicted')
            
            # 2. Residuals histogram
            axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            axes[0, 1].axvline(x=0, color='red', linestyle='--')
            axes[0, 1].set_xlabel('Residuals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Residuals Distribution')
            
            # 3. Q-Q plot
            from scipy import stats
            (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
            axes[1, 0].grid(True)
            
            # 4. Residuals vs observation order
            axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_xlabel('Observation Order')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals vs Order')
            
            plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Residuals analysis plot saved to {self.save_path / save_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Could not create residuals analysis plot: {e}")
            return None
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                              model_name: str, top_n: int = 20,
                              save_name: str = None) -> Optional[Any]:
        """Plot feature importance."""
        if save_name is None:
            save_name = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        
        try:
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True)[:top_n]
            
            features, importances = zip(*sorted_importance)
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
            
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='skyblue')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
            
            # Invert y-axis to have most important at top
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {self.save_path / save_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Could not create feature importance plot: {e}")
            return None
    
    def plot_learning_curve(self, train_scores: List[float], val_scores: List[float],
                          train_sizes: List[int], model_name: str,
                          save_name: str = None) -> Optional[Any]:
        """Plot learning curve."""
        if save_name is None:
            save_name = f"learning_curve_{model_name.replace(' ', '_').lower()}.png"
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Score')
            ax.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Score')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Score')
            ax.set_title(f'Learning Curve - {model_name}')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Learning curve plot saved to {self.save_path / save_name}")
            return fig
            
        except Exception as e:
            logger.error(f"Could not create learning curve plot: {e}")
            return None