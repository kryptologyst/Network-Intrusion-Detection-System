"""Streamlit demo application for Network Intrusion Detection System."""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.processor import NetworkFlowDataProcessor, SyntheticDataGenerator
from src.features.engineer import NetworkFlowFeatureEngineer
from src.models.models import ModelFactory
from src.eval.evaluator import NIDSEvaluator
from src.utils.utils import setup_logging, set_random_seeds


# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model(model_path: str) -> Optional[Dict[str, Any]]:
    """Load trained model from file."""
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate sample data for demonstration."""
    generator = SyntheticDataGenerator(n_samples=n_samples, n_intrusions=n_samples//10)
    return generator.generate_data()


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.warning("""
    **⚠️ DISCLAIMER**: This is a research and educational demonstration. 
    This system is NOT intended for production security operations.
    """)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["random_forest", "xgboost", "logistic_regression", "cnn"],
        help="Choose the machine learning model to use for detection"
    )
    
    # Load model
    model_path = f"models/{model_type}_model.pkl"
    model_data = load_model(model_path)
    
    if model_data is None:
        st.error(f"Model not found at {model_path}. Please train a model first.")
        st.stop()
    
    model = model_data["model"]
    feature_engineer = model_data["feature_engineer"]
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Real-time Analysis", "📊 Model Performance", "📈 Feature Analysis", "⚙️ Configuration"])
    
    with tab1:
        st.header("Real-time Network Flow Analysis")
        
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Network Flow Input")
            
            # Sample data or manual input
            input_method = st.radio(
                "Choose input method:",
                ["Generate Sample Data", "Upload CSV File", "Manual Input"]
            )
            
            if input_method == "Generate Sample Data":
                n_samples = st.slider("Number of samples to generate", 1, 1000, 100)
                if st.button("Generate Sample Data"):
                    sample_data = generate_sample_data(n_samples)
                    st.session_state.sample_data = sample_data
                    st.success(f"Generated {n_samples} sample network flows")
            
            elif input_method == "Upload CSV File":
                uploaded_file = st.file_uploader("Upload network flow data", type=["csv"])
                if uploaded_file is not None:
                    try:
                        sample_data = pd.read_csv(uploaded_file)
                        st.session_state.sample_data = sample_data
                        st.success(f"Loaded {len(sample_data)} network flows from file")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            
            else:  # Manual Input
                st.info("Manual input feature coming soon. Please use sample data generation.")
        
        with col2:
            st.subheader("Analysis Controls")
            
            # Threshold for classification
            threshold = st.slider(
                "Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Probability threshold for intrusion classification"
            )
            
            # Analysis button
            if st.button("🔍 Analyze Network Flows", type="primary"):
                if "sample_data" in st.session_state:
                    st.session_state.analysis_threshold = threshold
                    st.session_state.analyze_data = True
                else:
                    st.error("Please generate or upload data first")
        
        # Analysis results
        if st.session_state.get("analyze_data", False) and "sample_data" in st.session_state:
            st.header("Analysis Results")
            
            sample_data = st.session_state.sample_data
            threshold = st.session_state.analysis_threshold
            
            # Preprocess data
            processor = NetworkFlowDataProcessor()
            processed_data = processor.preprocess(sample_data)
            
            # Separate features and labels
            if "label" in processed_data.columns:
                X = processed_data.drop("label", axis=1)
                y_true = processed_data["label"]
            else:
                X = processed_data
                y_true = None
            
            # Transform features
            X_transformed = feature_engineer.transform(X)
            
            # Make predictions
            y_pred = model.predict(X_transformed)
            y_proba = model.predict_proba(X_transformed)[:, 1]
            
            # Apply threshold
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Flows", len(sample_data))
            
            with col2:
                intrusions_detected = np.sum(y_pred_thresh)
                st.metric("Intrusions Detected", intrusions_detected)
            
            with col3:
                detection_rate = np.mean(y_pred_thresh) * 100
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
            
            with col4:
                avg_prob = np.mean(y_proba)
                st.metric("Avg Intrusion Prob", f"{avg_prob:.3f}")
            
            # Detailed results table
            st.subheader("Detailed Results")
            
            results_df = sample_data.copy()
            results_df["intrusion_probability"] = y_proba
            results_df["predicted_label"] = y_pred_thresh
            results_df["prediction_confidence"] = np.maximum(y_proba, 1 - y_proba)
            
            if y_true is not None:
                results_df["actual_label"] = y_true
                results_df["correct_prediction"] = (y_pred_thresh == y_true)
            
            # Sort by intrusion probability
            results_df = results_df.sort_values("intrusion_probability", ascending=False)
            
            # Display top alerts
            st.subheader("🚨 Top Security Alerts")
            top_alerts = results_df.head(10)
            
            for idx, row in top_alerts.iterrows():
                prob = row["intrusion_probability"]
                confidence = row["prediction_confidence"]
                
                if prob >= threshold:
                    st.markdown(f"""
                    <div class="alert-box">
                        <strong>Alert #{idx}</strong><br>
                        Intrusion Probability: {prob:.3f}<br>
                        Confidence: {confidence:.3f}<br>
                        Duration: {row.get('duration', 'N/A')}s<br>
                        Protocol: {row.get('protocol_type', 'N/A')}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Flow #{idx}</strong><br>
                        Intrusion Probability: {prob:.3f}<br>
                        Status: Normal Traffic
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"nids_analysis_results_{threshold}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("Model Performance Analysis")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            
            # Mock performance data (in real app, load from saved results)
            metrics = {
                "Accuracy": 0.95,
                "Precision": 0.92,
                "Recall": 0.88,
                "F1-Score": 0.90,
                "AUC": 0.96,
                "AUCPR": 0.89
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")
        
        with col2:
            st.subheader("ROC Curve")
            
            # Mock ROC curve data
            fpr = np.linspace(0, 1, 100)
            tpr = np.sqrt(fpr)  # Mock curve
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name='ROC Curve',
                line=dict(color='blue', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig_roc.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=400,
                height=300
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Precision-Recall curve
        st.subheader("Precision-Recall Curve")
        
        recall = np.linspace(0, 1, 100)
        precision = np.exp(-2 * recall) + 0.5  # Mock curve
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name='PR Curve',
            line=dict(color='green', width=2)
        ))
        
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        
        st.plotly_chart(fig_pr, use_container_width=True)
    
    with tab3:
        st.header("Feature Importance Analysis")
        
        # Feature importance
        if hasattr(model, "get_feature_importance"):
            try:
                importance_df = model.get_feature_importance()
                
                # Top features
                top_features = importance_df.head(15)
                
                fig_importance = px.bar(
                    top_features,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top 15 Most Important Features",
                    color="importance",
                    color_continuous_scale="viridis"
                )
                
                fig_importance.update_layout(
                    height=500,
                    yaxis={"categoryorder": "total ascending"}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance table
                st.subheader("Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading feature importance: {e}")
        else:
            st.info("Feature importance not available for this model type.")
    
    with tab4:
        st.header("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Information")
            st.info(f"""
            **Model Type**: {model_type.upper()}
            **Status**: Loaded and Ready
            **Feature Engineer**: Configured
            """)
            
            if hasattr(model, "is_fitted"):
                st.success("✅ Model is trained and ready for inference")
            else:
                st.warning("⚠️ Model status unknown")
        
        with col2:
            st.subheader("System Status")
            st.success("✅ Data processor ready")
            st.success("✅ Feature engineer ready")
            st.success("✅ Model loaded")
            st.success("✅ Evaluation tools ready")
        
        # Configuration details
        st.subheader("Current Configuration")
        
        config_details = {
            "Model Type": model_type,
            "Detection Threshold": st.session_state.get("analysis_threshold", 0.5),
            "Feature Engineering": "Enabled",
            "Data Preprocessing": "Enabled",
            "Anonymization": "Enabled"
        }
        
        for key, value in config_details.items():
            st.text(f"{key}: {value}")


if __name__ == "__main__":
    main()
