"""
Streamlit App for Solar Panel Efficiency Prediction
Interactive demo interface for solar panel efficiency prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path so imports work
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Solar Panel Efficiency Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">☀️ Solar Panel Efficiency Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict solar panel efficiency based on environmental conditions</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Load model and feature info
@st.cache_resource
def load_model_and_features():
    """Load the trained model and feature information."""
    model_path = Path("results/models")
    
    # Try to find model files
    model_files = list(model_path.glob("*.joblib")) if model_path.exists() else []
    
    if not model_files:
        return None, None, None
    
    # Prefer random forest (best performer)
    preferred = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
    selected_model = None
    
    for pref in preferred:
        matching = [f for f in model_files if pref in f.name.lower()]
        if matching:
            selected_model = matching[0]
            break
    
    if not selected_model:
        selected_model = model_files[0]
    
    try:
        model = joblib.load(selected_model)
        model_name = selected_model.stem.replace('_', ' ').title()
        
        # Load feature info if available
        feature_info_path = Path("results/feature_info.json")
        if feature_info_path.exists():
            import json
            with open(feature_info_path) as f:
                feature_info = json.load(f)
            feature_names = feature_info.get('feature_names', [])
        else:
            # Fallback: try to get from model
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = None
        
        return model, model_name, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, model_name, feature_names = load_model_and_features()

if model is None:
    st.error("No trained model found. Please run `python train.py` first to train the model.")
    st.info("After training completes, the model will be saved in `results/models/` directory.")
    st.stop()

st.sidebar.success(f"Loaded: {model_name}")

# Display CV results and physics baseline
metrics_path = Path("results/metrics.json")
if metrics_path.exists():
    import json
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    best_model_key = metrics.get('best_model', '')
    
    # Find the best model metrics
    cv_results = metrics.get('cross_validation_results', {})
    final_eval = metrics.get('final_evaluation', {})
    physics_baseline = metrics.get('physics_baseline', {})
    
    if best_model_key in cv_results:
        st.write("---")
        st.subheader("Model Performance")
        
        # CV metrics in columns
        col1, col2, col3 = st.columns(3)
        
        cv_data = cv_results[best_model_key]
        with col1:
            st.metric(
                "MAE (CV)",
                f"{cv_data['mae_mean']:.4f}",
                delta=f"± {cv_data['mae_std']:.4f}",
                help="Mean Absolute Error from cross-validation"
            )
        with col2:
            st.metric(
                "RMSE (CV)",
                f"{cv_data['rmse_mean']:.4f}",
                delta=f"± {cv_data['rmse_std']:.4f}",
                help="Root Mean Squared Error from cross-validation"
            )
        with col3:
            st.metric(
                "R² (CV)",
                f"{cv_data['r2_mean']:.4f}",
                delta=f"± {cv_data['r2_std']:.4f}",
                help="R-squared score from cross-validation"
            )
        
        # Physics baseline comparison
        if physics_baseline and physics_baseline.get('rmse') is not None:
            baseline_rmse = physics_baseline['rmse']
            ml_rmse = cv_data['rmse_mean']
            improvement = ((baseline_rmse - ml_rmse) / baseline_rmse) * 100
            
            st.write("")
            st.info(
                f"**Physics Baseline:** RMSE = {baseline_rmse:.4f} | "
                f"**ML Improvement:** {improvement:.1f}% error reduction"
            )

# Display model info
with st.sidebar.expander("Model Information"):
    st.write(f"**Model Type:** {model_name}")
    if feature_names:
        st.write(f"**Features Used:** {len(feature_names)}")
    
    # Load metrics if available
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        best_model = metrics.get('best_model', model_name)
        if best_model.lower().replace(' ', '_') in model_name.lower():
            final_eval = metrics.get('final_evaluation', {})
            if best_model in final_eval:
                model_metrics = final_eval[best_model]['metrics']
                st.write(f"**R² Score:** {model_metrics.get('r2', 0):.4f}")
                st.write(f"**RMSE:** {model_metrics.get('rmse', 0):.4f}")
                st.write(f"**MAE:** {model_metrics.get('mae', 0):.4f}")

# Main content
st.write("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Make Predictions", "Batch Predictions", "About"])

with tab1:
    st.header("Single Prediction")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environmental Conditions")
        temperature = st.slider("Temperature (°C)", 
                               min_value=-10.0, max_value=50.0, value=25.0, step=0.5,
                               help="Ambient temperature in degrees Celsius")
        
        irradiance = st.slider("Irradiance (W/m²)", 
                              min_value=0.0, max_value=1200.0, value=800.0, step=10.0,
                              help="Solar irradiance in watts per square meter")
        
        humidity = st.slider("Humidity (%)", 
                            min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                            help="Relative humidity percentage")
        
        cloud_coverage = st.slider("Cloud Coverage (%)", 
                                   min_value=0.0, max_value=100.0, value=20.0, step=5.0,
                                   help="Percentage of sky covered by clouds")
    
    with col2:
        st.subheader("Panel Specifications")
        panel_age = st.slider("Panel Age (years)", 
                             min_value=0.0, max_value=30.0, value=5.0, step=0.5,
                             help="Age of the solar panel in years")
        
        maintenance_count = st.number_input("Maintenance Count", 
                                           min_value=0, max_value=100, value=5,
                                           help="Number of times panel has been maintained")
        
        soiling_ratio = st.slider("Soiling Ratio", 
                                 min_value=0.0, max_value=1.0, value=0.95, step=0.01,
                                 help="Cleanliness ratio (1.0 = perfectly clean)")
        
        voltage = st.slider("Voltage (V)", 
                          min_value=0.0, max_value=50.0, value=30.0, step=0.5,
                          help="Operating voltage in volts")
        
        current = st.slider("Current (A)", 
                          min_value=0.0, max_value=20.0, value=8.0, step=0.1,
                          help="Operating current in amperes")
    
    # Additional inputs
    with st.expander("⚙️ Additional Parameters"):
        col3, col4 = st.columns(2)
        with col3:
            module_temperature = st.slider("Module Temperature (°C)", 
                                         min_value=-10.0, max_value=80.0, value=35.0, step=0.5)
            wind_speed = st.slider("Wind Speed (m/s)", 
                                  min_value=0.0, max_value=30.0, value=3.0, step=0.5)
        with col4:
            pressure = st.slider("Atmospheric Pressure (hPa)", 
                               min_value=900.0, max_value=1100.0, value=1013.0, step=1.0)
            string_id = st.number_input("String ID", min_value=0, max_value=100, value=0)
    
    st.write("---")
    
    # Prediction button
    if st.button("Predict Efficiency", type="primary", use_container_width=True):
        # Create input dataframe
        # Note: We need to match the exact features the model was trained on
        # For simplicity, we'll create a basic feature set
        
        input_data = pd.DataFrame({
            'temperature': [temperature],
            'irradiance': [irradiance],
            'humidity': [humidity],
            'panel_age': [panel_age],
            'maintenance_count': [maintenance_count],
            'soiling_ratio': [soiling_ratio],
            'voltage': [voltage],
            'current': [current],
            'module_temperature': [module_temperature],
            'cloud_coverage': [cloud_coverage],
            'wind_speed': [wind_speed],
            'pressure': [pressure],
            'string_id': [string_id]
        })
        
        # If we know the exact features, align with them
        if feature_names:
            # Add missing features with default values
            for feature in feature_names:
                if feature not in input_data.columns:
                    input_data[feature] = 0
            # Reorder to match training
            input_data = input_data[feature_names]
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display result
            st.success("Prediction Complete!")
            
            # Create nice display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Efficiency", f"{prediction:.4f}", 
                         help="Predicted efficiency value (0-1 scale)")
            
            with col2:
                efficiency_pct = prediction * 100
                st.metric("Efficiency Percentage", f"{efficiency_pct:.2f}%",
                         help="Efficiency as percentage")
            
            with col3:
                # Simple rating
                if efficiency_pct >= 80:
                    rating = "Excellent ⭐⭐⭐⭐⭐"
                    color = "green"
                elif efficiency_pct >= 60:
                    rating = "Good ⭐⭐⭐⭐"
                    color = "blue"
                elif efficiency_pct >= 40:
                    rating = "Fair ⭐⭐⭐"
                    color = "orange"
                else:
                    rating = "Poor ⭐⭐"
                    color = "red"
                st.metric("Rating", rating)
            
            # Interpretation
            st.info(f"**Interpretation:** This configuration is predicted to operate at {efficiency_pct:.2f}% efficiency.")
            
            # Feature importance (if tree-based model)
            if hasattr(model, 'feature_importances_') and feature_names:
                with st.expander("Feature Contributions"):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 10 Most Important Features')
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.close()
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("This might be due to feature mismatch. Please ensure the model was trained successfully.")

with tab2:
    st.header("Batch Predictions")
    st.write("Upload a CSV file with multiple samples for batch prediction.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(data.head())
            
            if st.button("Predict for All Samples", type="primary"):
                # Align features
                if feature_names:
                    for feature in feature_names:
                        if feature not in data.columns:
                            data[feature] = 0
                    data_aligned = data[feature_names]
                else:
                    data_aligned = data
                
                try:
                    predictions = model.predict(data_aligned)
                    
                    result_df = data.copy()
                    result_df['Predicted_Efficiency'] = predictions
                    result_df['Efficiency_Percentage'] = predictions * 100
                    
                    st.success(f"Predicted {len(predictions)} samples!")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics
                    st.write("**Prediction Statistics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean Efficiency", f"{predictions.mean():.4f}")
                    col2.metric("Std Dev", f"{predictions.std():.4f}")
                    col3.metric("Min", f"{predictions.min():.4f}")
                    col4.metric("Max", f"{predictions.max():.4f}")
                    
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### 🌟 Solar Panel Efficiency Predictor
    
    This application uses machine learning to predict solar panel efficiency based on environmental 
    conditions and panel characteristics.
    
    #### Model Information
    - **Algorithm:** Random Forest / Gradient Boosting (ensemble method)
    - **Features:** Environmental conditions, panel specifications, and derived features
    - **Performance:** Trained on 20,000+ samples with cross-validation
    
    #### How to Use
    1. **Single Prediction:** Adjust sliders to set conditions, click "Predict Efficiency"
    2. **Batch Prediction:** Upload a CSV file with multiple samples
    
    #### Input Features
    - **Environmental:** Temperature, irradiance, humidity, cloud coverage, wind speed, pressure
    - **Panel Specs:** Age, maintenance history, soiling ratio
    - **Electrical:** Voltage, current, module temperature
    
    #### Model Training
    The model was trained using:
    - Feature engineering (physics-based features)
    - Feature selection (18 most important features)
    - 5-fold cross-validation
    - Bootstrap uncertainty quantification
    - SHAP analysis for interpretability
    
    #### Tips for Best Results
    - Typical operating conditions: 20-35°C, 600-1000 W/m²
    - Regular maintenance improves efficiency
    - Clean panels (high soiling ratio) perform better
    - Check feature importance to understand key factors
    
    #### Notes
    - Predictions are based on historical data patterns
    - Always validate with actual measurements
    - For production use, consider regular model updates
    
    ---
    
    **Version:** 1.0  
    **Last Updated:** 2025-10-28
    """)
    
    # Display results summary if available
    if Path("results/model_comparison.csv").exists():
        st.write("### Model Performance Summary")
        comparison_df = pd.read_csv("results/model_comparison.csv")
        st.dataframe(comparison_df)
    
    # What-if analysis static plots
    st.write("### What-If Analysis (Static)")
    
    temp_plot_path = Path("results/plots/efficiency_vs_temperature.png")
    irrad_plot_path = Path("results/plots/whatif_irradiance.png")
    
    if temp_plot_path.exists():
        st.write("**Efficiency vs Temperature**")
        st.image(str(temp_plot_path), use_container_width=True)
    
    if irrad_plot_path.exists():
        st.write("**Efficiency vs Irradiance**")
        st.image(str(irrad_plot_path), use_container_width=True)
    
    if not temp_plot_path.exists() and not irrad_plot_path.exists():
        st.info("Run the training pipeline to generate what-if analysis plots.")
    
    ci_plot_path = Path("results/plots/predictions_with_ci.png")
    
    if ci_plot_path.exists():
        st.write("**Predictions with 95% Confidence Intervals**")
        st.image(str(ci_plot_path), use_container_width=True)
    else:
        st.info("Run the training pipeline to generate analysis plots.")

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>Solar Panel Efficiency Predictor | Built with Streamlit</p>
    <p>⚡ Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)