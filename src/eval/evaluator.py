"""Evaluation metrics and utilities for Network Intrusion Detection System."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class NIDSEvaluator:
    """Evaluator for Network Intrusion Detection System."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize evaluator.

        Args:
            logger: Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        k_values: List[int] = [10, 50, 100],
        target_tpr: float = 0.95,
    ) -> Dict[str, Any]:
        """Evaluate model performance comprehensively.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            k_values: List of K values for precision@K.
            target_tpr: Target true positive rate for FPR calculation.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.logger.info("Evaluating model performance")

        metrics = {}

        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))

        # Probability-based metrics
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))

        # Precision@K metrics
        if y_proba is not None:
            metrics.update(self._calculate_precision_at_k(y_true, y_proba, k_values))

        # FPR at target TPR
        if y_proba is not None:
            metrics.update(self._calculate_fpr_at_tpr(y_true, y_proba, target_tpr))

        # Operational metrics
        metrics.update(self._calculate_operational_metrics(y_true, y_pred))

        self.logger.info("Model evaluation completed")
        return metrics

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of basic metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1_score": self._calculate_f1_score(y_true, y_pred),
            "specificity": self._calculate_specificity(y_true, y_pred),
            "balanced_accuracy": self._calculate_balanced_accuracy(y_true, y_pred),
        }

    def _calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based metrics.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary of probability metrics.
        """
        return {
            "auc": roc_auc_score(y_true, y_proba),
            "aucpr": average_precision_score(y_true, y_proba),
        }

    def _calculate_precision_at_k(self, y_true: np.ndarray, y_proba: np.ndarray, k_values: List[int]) -> Dict[str, float]:
        """Calculate precision@K metrics.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            k_values: List of K values.

        Returns:
            Dictionary of precision@K metrics.
        """
        metrics = {}
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(y_proba)[::-1]
        sorted_y_true = y_true[sorted_indices]
        
        for k in k_values:
            if k <= len(sorted_y_true):
                top_k_true = sorted_y_true[:k]
                precision_k = np.sum(top_k_true) / k if k > 0 else 0.0
                metrics[f"precision_at_{k}"] = precision_k
        
        return metrics

    def _calculate_fpr_at_tpr(self, y_true: np.ndarray, y_proba: np.ndarray, target_tpr: float) -> Dict[str, float]:
        """Calculate FPR at target TPR.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            target_tpr: Target true positive rate.

        Returns:
            Dictionary with FPR at target TPR.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        
        # Find threshold closest to target TPR
        tpr_diff = np.abs(tpr - target_tpr)
        closest_idx = np.argmin(tpr_diff)
        
        fpr_at_target_tpr = fpr[closest_idx]
        actual_tpr = tpr[closest_idx]
        threshold = thresholds[closest_idx]
        
        return {
            f"fpr_at_tpr_{target_tpr}": fpr_at_target_tpr,
            f"actual_tpr_at_threshold": actual_tpr,
            f"threshold_for_tpr_{target_tpr}": threshold,
        }

    def _calculate_operational_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate operational metrics for security applications.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of operational metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_samples = len(y_true)
        total_positives = np.sum(y_true)
        total_negatives = total_samples - total_positives
        
        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "alert_rate": (tp + fp) / total_samples if total_samples > 0 else 0.0,
            "false_alarm_rate": fp / total_negatives if total_negatives > 0 else 0.0,
            "detection_rate": tp / total_positives if total_positives > 0 else 0.0,
            "alert_volume_per_1000": (tp + fp) * 1000 / total_samples if total_samples > 0 else 0.0,
        }

    def _calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="binary", zero_division=0)

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (recall + specificity) / 2.0

    def calculate_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate precision-recall curve.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Tuple of (precision, recall, thresholds).
        """
        return precision_recall_curve(y_true, y_proba)

    def calculate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Tuple of (fpr, tpr, thresholds).
        """
        return roc_curve(y_true, y_proba)

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate detailed classification report.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Classification report string.
        """
        return classification_report(y_true, y_pred, target_names=["Normal", "Intrusion"])

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)

    def evaluate_threshold_performance(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Evaluate performance at different thresholds.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            thresholds: List of thresholds to evaluate. If None, uses default range.

        Returns:
            DataFrame with threshold performance metrics.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()

        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            metrics = self._calculate_basic_metrics(y_true, y_pred_thresh)
            metrics["threshold"] = threshold
            
            results.append(metrics)
        
        return pd.DataFrame(results)

    def calculate_cost_benefit_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cost_fp: float = 1.0,
        cost_fn: float = 10.0,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Calculate cost-benefit analysis for different thresholds.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            cost_fp: Cost of false positive.
            cost_fn: Cost of false negative.
            thresholds: List of thresholds to evaluate.

        Returns:
            DataFrame with cost-benefit analysis.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9).tolist()

        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            total_benefit = tp  # Assuming benefit of 1 for each true positive
            
            results.append({
                "threshold": threshold,
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "total_cost": total_cost,
                "total_benefit": total_benefit,
                "net_benefit": total_benefit - total_cost,
                "cost_per_detection": total_cost / tp if tp > 0 else float("inf"),
            })
        
        return pd.DataFrame(results)

    def calculate_model_comparison(
        self,
        models_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare multiple models' performance.

        Args:
            models_results: Dictionary with model names as keys and evaluation results as values.

        Returns:
            DataFrame with model comparison.
        """
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {"model": model_name}
            row.update(results)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values("aucpr", ascending=False)

    def generate_evaluation_summary(
        self,
        metrics: Dict[str, Any],
        model_name: str = "Model",
    ) -> str:
        """Generate a summary of evaluation results.

        Args:
            metrics: Dictionary of evaluation metrics.
            model_name: Name of the model.

        Returns:
            Formatted evaluation summary.
        """
        summary = f"\n{'='*50}\n"
        summary += f"EVALUATION SUMMARY: {model_name}\n"
        summary += f"{'='*50}\n\n"
        
        # Basic metrics
        summary += "BASIC METRICS:\n"
        summary += f"  Accuracy:     {metrics.get('accuracy', 0):.4f}\n"
        summary += f"  Precision:    {metrics.get('precision', 0):.4f}\n"
        summary += f"  Recall:       {metrics.get('recall', 0):.4f}\n"
        summary += f"  F1-Score:     {metrics.get('f1_score', 0):.4f}\n"
        summary += f"  Specificity:  {metrics.get('specificity', 0):.4f}\n\n"
        
        # Probability metrics
        if "auc" in metrics:
            summary += "PROBABILITY METRICS:\n"
            summary += f"  AUC:          {metrics.get('auc', 0):.4f}\n"
            summary += f"  AUCPR:        {metrics.get('aucpr', 0):.4f}\n\n"
        
        # Precision@K metrics
        precision_k_metrics = {k: v for k, v in metrics.items() if k.startswith("precision_at_")}
        if precision_k_metrics:
            summary += "PRECISION@K METRICS:\n"
            for metric_name, value in precision_k_metrics.items():
                summary += f"  {metric_name.replace('_', '@').title()}: {value:.4f}\n"
            summary += "\n"
        
        # Operational metrics
        summary += "OPERATIONAL METRICS:\n"
        summary += f"  Alert Rate:           {metrics.get('alert_rate', 0):.4f}\n"
        summary += f"  False Alarm Rate:     {metrics.get('false_alarm_rate', 0):.4f}\n"
        summary += f"  Detection Rate:       {metrics.get('detection_rate', 0):.4f}\n"
        summary += f"  Alert Volume/1000:    {metrics.get('alert_volume_per_1000', 0):.2f}\n"
        
        summary += f"\n{'='*50}\n"
        
        return summary
