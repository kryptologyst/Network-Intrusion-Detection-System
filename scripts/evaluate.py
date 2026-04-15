#!/usr/bin/env python3
"""Script to evaluate trained models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml
import pickle

from src.data.processor import NetworkFlowDataProcessor
from src.features.engineer import NetworkFlowFeatureEngineer
from src.eval.evaluator import NIDSEvaluator
from src.utils.utils import setup_logging, Timer


def main():
    """Main function for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained network intrusion detection models")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--test-data", type=str, default="data/test.parquet",
                       help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for evaluation results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting model evaluation")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Test data: {args.test_data}")
    
    with Timer("Model Loading"):
        # Load trained model
        try:
            with open(args.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            model = model_data["model"]
            feature_engineer = model_data["feature_engineer"]
            config = model_data["config"]
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    with Timer("Data Loading"):
        # Load test data
        try:
            processor = NetworkFlowDataProcessor()
            test_data = processor.load_data(args.test_data)
            test_data = processor.preprocess(test_data)
            
            # Separate features and labels
            X_test = test_data.drop("label", axis=1)
            y_test = test_data["label"]
            
            logger.info(f"Test data loaded: {len(test_data)} samples")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return
    
    with Timer("Feature Transformation"):
        # Transform features
        X_test_transformed = feature_engineer.transform(X_test)
        logger.info(f"Features transformed: {X_test_transformed.shape}")
    
    with Timer("Model Evaluation"):
        # Initialize evaluator
        evaluator = NIDSEvaluator(logger=logger)
        
        # Get predictions
        y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)[:, 1]
        
        # Evaluate model
        metrics = evaluator.evaluate_model(y_test, y_pred, y_proba)
        
        # Generate detailed evaluation report
        classification_report = evaluator.generate_classification_report(y_test, y_pred)
        confusion_matrix = evaluator.calculate_confusion_matrix(y_test, y_pred)
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(evaluator.generate_evaluation_summary(metrics, "Model"))
        
        print("\nClassification Report:")
        print(classification_report)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix)
    
    with Timer("Results Saving"):
        # Save evaluation results
        results = {
            "metrics": metrics,
            "classification_report": classification_report,
            "confusion_matrix": confusion_matrix.tolist(),
            "test_samples": len(y_test),
            "model_path": args.model_path,
            "test_data_path": args.test_data
        }
        
        results_path = output_dir / "evaluation_results.yaml"
        with open(results_path, "w") as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Evaluation results saved to: {results_path}")
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
