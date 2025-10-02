# src/features/engineering.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)

class FeatureEngineer:
    """Creates domain-specific features for solar panel efficiency prediction."""
    
    def __init__(self):
        self.config = get_config()
        self.feature_names = []
        self.poly_features = None
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature engineering pipeline.
        
        Solar panel efficiency is influenced by multiple physical factors:
        - Temperature coefficient affects performance (typically -0.4%/°C)
        - Irradiance directly correlates with power output
        - Humidity can affect electrical properties
        - Soiling reduces light absorption
        """
        logger.info("Starting feature engineering process")
        
        df_engineered = df.copy()
        
        # Handle missing values first
        df_engineered = self._handle_missing_values(df_engineered)
        
        # Solar-specific derived features
        df_engineered = self._create_solar_features(df_engineered)
        
        # Physics-based interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # Environmental composite features
        df_engineered = self._create_environmental_features(df_engineered)
        
        # Maintenance and aging features
        df_engineered = self._create_maintenance_features(df_engineered)
        
        # Performance ratios and indices
        df_engineered = self._create_performance_features(df_engineered)
        
        # Polynomial features for key relationships (with proper missing value handling)
        df_engineered = self._create_polynomial_features(df_engineered)
        
        logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
        logger.info(f"Added {len(self.feature_names)} new features")
        
        return df_engineered
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numerical columns before feature engineering."""
        numerical_cols = ['temperature', 'irradiance', 'humidity', 'panel_age', 
                         'maintenance_count', 'soiling_ratio', 'voltage', 'current', 
                         'module_temperature', 'cloud_coverage', 'wind_speed', 'pressure']
        
        # Only process columns that exist in the dataframe
        available_numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if available_numerical_cols:
            if not self.is_fitted:
                # Fit the imputer on first call (training data)
                df[available_numerical_cols] = self.numerical_imputer.fit_transform(df[available_numerical_cols])
                self.is_fitted = True
            else:
                # Transform using fitted imputer (test data)
                df[available_numerical_cols] = self.numerical_imputer.transform(df[available_numerical_cols])
        
        return df
    
    def _create_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create solar panel physics-based features."""
        
        # Temperature-corrected efficiency potential
        # Standard Test Conditions (STC): 25°C
        if 'temperature' in df.columns and 'module_temperature' in df.columns:
            df['temp_deviation_stc'] = df['module_temperature'] - 25
            df['temp_correction_factor'] = 1 - (df['temp_deviation_stc'] * 0.004)  # -0.4%/°C
            self.feature_names.extend(['temp_deviation_stc', 'temp_correction_factor'])
        
        # Irradiance performance ratio
        # STC irradiance: 1000 W/m²
        if 'irradiance' in df.columns:
            df['irradiance_ratio'] = df['irradiance'] / 1000
            df['irradiance_squared'] = np.where(df['irradiance'] > 0, df['irradiance'] ** 2, 0)
            self.feature_names.extend(['irradiance_ratio', 'irradiance_squared'])
        
        # Electrical power estimation (P = V × I)
        if 'voltage' in df.columns and 'current' in df.columns:
            df['power_output'] = df['voltage'] * df['current']
            df['power_density'] = np.where(df['irradiance'] > 0, 
                                         df['power_output'] / df['irradiance'], 0)
            self.feature_names.extend(['power_output', 'power_density'])
        
        # Soiling impact on available irradiance
        if 'soiling_ratio' in df.columns and 'irradiance' in df.columns:
            df['effective_irradiance'] = df['irradiance'] * (1 - df['soiling_ratio'])
            self.feature_names.append('effective_irradiance')
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features based on solar panel physics."""
        
        # Temperature × Irradiance (critical interaction)
        if 'temperature' in df.columns and 'irradiance' in df.columns:
            df['temp_irradiance_interaction'] = df['temperature'] * df['irradiance']
            self.feature_names.append('temp_irradiance_interaction')
        
        # Humidity × Temperature (affects electrical properties)
        if 'humidity' in df.columns and 'temperature' in df.columns:
            df['humidity_temp_product'] = df['humidity'] * df['temperature']
            df['humidity_temp_ratio'] = np.where(df['temperature'] > -273, 
                                               df['humidity'] / (df['temperature'] + 273.15), 0)
            self.feature_names.extend(['humidity_temp_product', 'humidity_temp_ratio'])
        
        # Wind cooling effect
        if 'wind_speed' in df.columns and 'module_temperature' in df.columns:
            df['wind_cooling_factor'] = np.where(df['module_temperature'] > 0,
                                               df['wind_speed'] / df['module_temperature'], 0)
            self.feature_names.append('wind_cooling_factor')
        
        # Cloud coverage × Irradiance
        if 'cloud_coverage' in df.columns and 'irradiance' in df.columns:
            df['clear_sky_factor'] = (1 - df['cloud_coverage']) * df['irradiance']
            self.feature_names.append('clear_sky_factor')
        
        return df
    
    def _create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite environmental indices."""
        
        # Overall environmental stress index
        stress_components = []
        weights = []
        
        if 'temperature' in df.columns:
            # High temperature stress (exponential above 25°C)
            temp_stress = np.where(df['temperature'] > 25, 
                                 np.exp((df['temperature'] - 25) * 0.05), 0)
            df['temperature_stress'] = temp_stress
            stress_components.append(temp_stress)
            weights.append(0.3)
            self.feature_names.append('temperature_stress')
        
        if 'humidity' in df.columns:
            # High humidity stress
            humidity_stress = np.where(df['humidity'] > 60, 
                                     (df['humidity'] - 60) / 40, 0)
            df['humidity_stress'] = humidity_stress
            stress_components.append(humidity_stress)
            weights.append(0.2)
            self.feature_names.append('humidity_stress')
        
        if 'soiling_ratio' in df.columns:
            # Soiling stress
            stress_components.append(df['soiling_ratio'])
            weights.append(0.25)
        
        if 'cloud_coverage' in df.columns:
            # Cloud coverage impact
            stress_components.append(df['cloud_coverage'])
            weights.append(0.25)
        
        # Calculate composite environmental stress
        if stress_components:
            weights = np.array(weights) / sum(weights)  # Normalize weights
            df['environmental_stress'] = sum(w * comp for w, comp in zip(weights, stress_components))
            self.feature_names.append('environmental_stress')
        
        # Atmospheric pressure effects (affects air density and cooling)
        if 'pressure' in df.columns:
            df['pressure_normalized'] = (df['pressure'] - 1013.25) / 1013.25  # Normalize to sea level
            self.feature_names.append('pressure_normalized')
        
        return df
    
    def _create_maintenance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create maintenance and aging-related features."""
        
        if 'panel_age' in df.columns and 'maintenance_count' in df.columns:
            # Maintenance frequency relative to age
            df['maintenance_per_year'] = np.where(df['panel_age'] > 0,
                                                df['maintenance_count'] / df['panel_age'], 0)
            
            # Degradation estimation (typical 0.5-0.8% per year)
            df['expected_degradation'] = df['panel_age'] * 0.006  # 0.6% per year
            
            # Maintenance effectiveness proxy
            df['maintenance_effectiveness'] = np.where(df['panel_age'] > 0,
                                                     df['maintenance_count'] / df['panel_age'], 0)
            
            self.feature_names.extend(['maintenance_per_year', 'expected_degradation', 'maintenance_effectiveness'])
        
        # Age-related performance decline
        if 'panel_age' in df.columns:
            df['age_squared'] = df['panel_age'] ** 2  # Non-linear aging effects
            df['age_log'] = np.log1p(df['panel_age'])  # Logarithmic aging
            self.feature_names.extend(['age_squared', 'age_log'])
        
        return df
    
    def _create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create performance ratios and indices."""
        
        # Voltage-current performance ratios
        if 'voltage' in df.columns and 'current' in df.columns:
            df['v_to_i_ratio'] = np.where(df['current'] != 0, df['voltage'] / df['current'], 0)
            
            # Normalized electrical parameters
            if 'irradiance' in df.columns:
                df['voltage_per_irradiance'] = np.where(df['irradiance'] > 0,
                                                       df['voltage'] / df['irradiance'], 0)
                df['current_per_irradiance'] = np.where(df['irradiance'] > 0,
                                                       df['current'] / df['irradiance'], 0)
                self.feature_names.extend(['voltage_per_irradiance', 'current_per_irradiance'])
            
            self.feature_names.append('v_to_i_ratio')
        
        # Module temperature vs ambient temperature difference
        if 'module_temperature' in df.columns and 'temperature' in df.columns:
            df['temp_differential'] = df['module_temperature'] - df['temperature']
            df['temp_differential_squared'] = df['temp_differential'] ** 2
            self.feature_names.extend(['temp_differential', 'temp_differential_squared'])
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for key relationships with proper fitting."""
    
        # Select key features for polynomial transformation
        key_features = []
        available_features = ['temperature', 'irradiance', 'humidity', 'voltage', 'current']
    
        for feat in available_features:
            if feat in df.columns:
                key_features.append(feat)
    
        if len(key_features) >= 2:
            try:
                # Always create a new PolynomialFeatures instance
                poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
                # Select subset for polynomial features
                subset_df = df[key_features].copy()
            
                # Ensure no missing values
                if subset_df.isnull().any().any():
                    logger.warning("Found NaN values in polynomial feature subset. Filling with median.")
                    subset_df = subset_df.fillna(subset_df.median())
            
                # Create polynomial features
                poly_data = poly_features.fit_transform(subset_df)
                poly_feature_names = poly_features.get_feature_names_out(key_features)
            
                # Add only interaction terms (skip original features)
                interaction_indices = [i for i, name in enumerate(poly_feature_names) 
                                    if '*' in name or '^2' in name]
            
                for idx in interaction_indices:
                    feature_name = f"poly_{poly_feature_names[idx].replace(' ', '_').replace('*', '_x_').replace('^', '_pow_')}"
                    df[feature_name] = poly_data[:, idx]
                    self.feature_names.append(feature_name)
                
            except Exception as e:
                logger.warning(f"Could not create polynomial features: {e}")
    
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        return self.feature_names.copy()