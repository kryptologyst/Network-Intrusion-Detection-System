"""Tests for Network Intrusion Detection System."""

import pytest
import numpy as np
import pandas as pd

from src.data.processor import NetworkFlowDataProcessor, SyntheticDataGenerator
from src.features.engineer import NetworkFlowFeatureEngineer
from src.models.models import ModelFactory
from src.eval.evaluator import NIDSEvaluator
from src.utils.utils import set_random_seeds, get_device


class TestSyntheticDataGenerator:
    """Test synthetic data generation."""

    def test_generate_data(self):
        """Test synthetic data generation."""
        generator = SyntheticDataGenerator(n_samples=100, n_intrusions=10)
        data = generator.generate_data()
        
        assert len(data) == 100
        assert data["label"].sum() == 10
        assert "duration" in data.columns
        assert "protocol_type" in data.columns
        assert "src_bytes" in data.columns
        assert "dst_bytes" in data.columns

    def test_data_distribution(self):
        """Test data distribution."""
        generator = SyntheticDataGenerator(n_samples=1000, n_intrusions=100)
        data = generator.generate_data()
        
        # Check that intrusions have different patterns than normal traffic
        normal_data = data[data["label"] == 0]
        intrusion_data = data[data["label"] == 1]
        
        # Intrusions should have higher average duration
        assert intrusion_data["duration"].mean() > normal_data["duration"].mean()
        
        # Intrusions should have higher average bytes
        assert intrusion_data["src_bytes"].mean() > normal_data["src_bytes"].mean()


class TestDataProcessor:
    """Test data processing."""

    def test_preprocessing(self):
        """Test data preprocessing."""
        processor = NetworkFlowDataProcessor()
        
        # Create sample data
        data = pd.DataFrame({
            "duration": [10, 20, 30],
            "protocol_type": [0, 1, 0],
            "src_bytes": [1000, 2000, 3000],
            "dst_bytes": [500, 1000, 1500],
            "label": [0, 1, 0]
        })
        
        processed = processor.preprocess(data)
        
        assert len(processed) == len(data)
        assert "duration" in processed.columns
        assert "protocol_type" in processed.columns

    def test_data_splitting(self):
        """Test data splitting."""
        processor = NetworkFlowDataProcessor()
        
        # Create sample data
        data = pd.DataFrame({
            "duration": np.random.exponential(10, 100),
            "protocol_type": np.random.choice([0, 1, 2], 100),
            "src_bytes": np.random.lognormal(6, 2, 100),
            "dst_bytes": np.random.lognormal(6, 2, 100),
            "label": np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(data)
        
        assert len(X_train) + len(X_val) + len(X_test) == len(data)
        assert len(y_train) + len(y_val) + len(y_test) == len(data)


class TestFeatureEngineer:
    """Test feature engineering."""

    def test_feature_engineering(self):
        """Test feature engineering."""
        engineer = NetworkFlowFeatureEngineer()
        
        # Create sample data
        data = pd.DataFrame({
            "duration": [10, 20, 30],
            "protocol_type": ["tcp", "udp", "tcp"],
            "service": ["http", "ftp", "http"],
            "flag": ["SF", "S0", "SF"],
            "src_bytes": [1000, 2000, 3000],
            "dst_bytes": [500, 1000, 1500]
        })
        
        # Engineer statistical features
        enhanced = engineer.engineer_statistical_features(data)
        
        assert "total_bytes" in enhanced.columns
        assert "byte_ratio" in enhanced.columns
        assert "byte_entropy" in enhanced.columns

    def test_fit_transform(self):
        """Test fit and transform."""
        engineer = NetworkFlowFeatureEngineer()
        
        # Create sample data
        data = pd.DataFrame({
            "duration": [10, 20, 30, 40],
            "protocol_type": ["tcp", "udp", "tcp", "udp"],
            "src_bytes": [1000, 2000, 3000, 4000],
            "dst_bytes": [500, 1000, 1500, 2000]
        })
        
        transformed = engineer.fit_transform(data)
        
        assert len(transformed) == len(data)
        assert transformed.shape[1] >= data.shape[1]


class TestModels:
    """Test model implementations."""

    def test_random_forest(self):
        """Test Random Forest model."""
        model = ModelFactory.create_model("random_forest", n_estimators=10)
        
        # Create sample data
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(y)
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 2

    def test_xgboost(self):
        """Test XGBoost model."""
        model = ModelFactory.create_model("xgboost", n_estimators=10)
        
        # Create sample data
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(y)
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 2

    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        model = ModelFactory.create_model("logistic_regression")
        
        # Create sample data
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        assert len(predictions) == len(y)
        assert probabilities.shape[0] == len(y)
        assert probabilities.shape[1] == 2


class TestEvaluator:
    """Test evaluation metrics."""

    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        evaluator = NIDSEvaluator()
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.8, 0.2, 0.9])
        
        metrics = evaluator.evaluate_model(y_true, y_pred, y_proba)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "auc" in metrics
        assert "aucpr" in metrics

    def test_precision_at_k(self):
        """Test precision@K calculation."""
        evaluator = NIDSEvaluator()
        
        # Create sample data with known precision@K
        y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        
        metrics = evaluator.evaluate_model(y_true, y_true, y_proba, k_values=[3])
        
        # Precision@3 should be 1.0 (all top 3 are positive)
        assert metrics["precision_at_3"] == 1.0


class TestUtils:
    """Test utility functions."""

    def test_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        
        # Generate some random numbers
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        
        # Reset seed and generate again
        set_random_seeds(42)
        rand3 = np.random.rand()
        rand4 = np.random.rand()
        
        # Should be the same
        assert rand1 == rand3
        assert rand2 == rand4

    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        # Should return a valid device
        assert device is not None
        assert hasattr(device, "type")


if __name__ == "__main__":
    pytest.main([__file__])
