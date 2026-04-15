#!/usr/bin/env python3
"""Script to train models for network intrusion detection."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from src.data.processor import NetworkFlowDataProcessor
from src.features.engineer import NetworkFlowFeatureEngineer
from src.models.models import ModelFactory
from src.eval.evaluator import NIDSEvaluator
from src.utils.utils import setup_logging, set_random_seeds, ensure_dir, Timer
from src.utils.config import Config


def main():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description="Train network intrusion detection models")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml",
                       help="Configuration file path")
    parser.add_argument("--data-path", type=str, default="data/synthetic_flows.parquet",
                       help="Path to training data")
    parser.add_argument("--model-type", type=str, default="random_forest",
                       choices=["random_forest", "xgboost", "logistic_regression", "cnn"],
                       help="Type of model to train")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for trained models")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging.level", "INFO")
    logger = setup_logging(level=log_level)
    
    # Set random seeds
    random_seed = config.get("data.random_seed", 42)
    set_random_seeds(random_seed)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    logger.info("Starting model training")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Data path: {args.data_path}")
    
    with Timer("Data Loading and Preprocessing"):
        # Load and preprocess data
        processor = NetworkFlowDataProcessor(
            anonymize_ips=config.get("data.anonymize_ips", True),
            ip_columns=config.get("data.ip_columns", ["src_ip", "dst_ip"]),
            logger=logger
        )
        
        data = processor.load_data(args.data_path)
        data = processor.preprocess(data)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            data,
            target_col="label",
            test_size=config.get("data.test_split", 0.2),
            val_size=config.get("data.val_split", 0.2),
            random_state=random_seed
        )
    
    with Timer("Feature Engineering"):
        # Feature engineering
        feature_engineer = NetworkFlowFeatureEngineer(
            categorical_columns=config.get("features.categorical_columns"),
            numerical_columns=config.get("features.numerical_columns"),
            scaler_type=config.get("features.scaler_type", "robust"),
            logger=logger
        )
        
        # Engineer additional features
        if config.get("features.engineer_statistical", True):
            X_train = feature_engineer.engineer_statistical_features(X_train)
            X_val = feature_engineer.engineer_statistical_features(X_val)
            X_test = feature_engineer.engineer_statistical_features(X_test)
        
        if config.get("features.engineer_network", True):
            X_train = feature_engineer.engineer_network_features(X_train)
            X_val = feature_engineer.engineer_network_features(X_val)
            X_test = feature_engineer.engineer_network_features(X_test)
        
        if config.get("features.engineer_behavioral", True):
            X_train = feature_engineer.engineer_behavioral_features(X_train)
            X_val = feature_engineer.engineer_behavioral_features(X_val)
            X_test = feature_engineer.engineer_behavioral_features(X_test)
        
        # Fit and transform features
        X_train_transformed = feature_engineer.fit_transform(X_train)
        X_val_transformed = feature_engineer.transform(X_val)
        X_test_transformed = feature_engineer.transform(X_test)
    
    with Timer("Model Training"):
        # Get model configuration
        model_config = config.get(f"models.{args.model_type}", {})
        
        # Create model
        if args.model_type == "cnn":
            model_config["input_dim"] = X_train_transformed.shape[1]
        
        model = ModelFactory.create_model(args.model_type, **model_config)
        
        # Train model
        if args.model_type == "cnn":
            model.fit(
                X_train_transformed,
                y_train,
                X_val=X_val_transformed,
                y_val=y_val,
                epochs=config.get("training.epochs", 100),
                batch_size=config.get("training.batch_size", 32),
                early_stopping_patience=config.get("training.early_stopping_patience", 10)
            )
        else:
            model.fit(X_train_transformed, y_train)
    
    with Timer("Model Evaluation"):
        # Evaluate model
        evaluator = NIDSEvaluator(logger=logger)
        
        # Get predictions
        y_pred_train = model.predict(X_train_transformed)
        y_pred_val = model.predict(X_val_transformed)
        y_pred_test = model.predict(X_test_transformed)
        
        y_proba_train = model.predict_proba(X_train_transformed)[:, 1]
        y_proba_val = model.predict_proba(X_val_transformed)[:, 1]
        y_proba_test = model.predict_proba(X_test_transformed)[:, 1]
        
        # Evaluate on different sets
        train_metrics = evaluator.evaluate_model(y_train, y_pred_train, y_proba_train)
        val_metrics = evaluator.evaluate_model(y_val, y_pred_val, y_proba_val)
        test_metrics = evaluator.evaluate_model(y_test, y_pred_test, y_proba_test)
        
        # Print results
        print("\n" + "="*60)
        print(f"MODEL TRAINING RESULTS: {args.model_type.upper()}")
        print("="*60)
        
        print("\nTRAINING SET:")
        print(evaluator.generate_evaluation_summary(train_metrics, f"{args.model_type}_train"))
        
        print("\nVALIDATION SET:")
        print(evaluator.generate_evaluation_summary(val_metrics, f"{args.model_type}_val"))
        
        print("\nTEST SET:")
        print(evaluator.generate_evaluation_summary(test_metrics, f"{args.model_type}_test"))
    
    with Timer("Model Saving"):
        # Save model and results
        model_path = output_dir / f"{args.model_type}_model.pkl"
        results_path = output_dir / f"{args.model_type}_results.yaml"
        
        # Save model (simplified - in practice, use proper serialization)
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_engineer": feature_engineer,
                "config": config._config
            }, f)
        
        # Save results
        results = {
            "model_type": args.model_type,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "config": dict(config._config)
        }
        
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
