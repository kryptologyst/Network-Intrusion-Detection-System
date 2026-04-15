"""Utility functions for Network Intrusion Detection System."""

import hashlib
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, logs to console only.
        log_format: Custom log format string.

    Returns:
        Configured logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("nids")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).

    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device


def hash_ip_address(ip: str) -> str:
    """Hash IP address for privacy protection.

    Args:
        ip: IP address string.

    Returns:
        SHA-256 hash of the IP address.
    """
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def anonymize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Anonymize sensitive columns in DataFrame.

    Args:
        df: Input DataFrame.
        columns: List of column names to anonymize.

    Returns:
        DataFrame with anonymized columns.
    """
    df_anon = df.copy()
    
    for col in columns:
        if col in df_anon.columns:
            df_anon[col] = df_anon[col].astype(str).apply(hash_ip_address)
    
    return df_anon


def create_time_based_splits(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create time-based train/validation/test splits.

    Args:
        df: Input DataFrame with time column.
        time_col: Name of the time column.
        train_ratio: Ratio of data for training.
        val_ratio: Ratio of data for validation.
        test_ratio: Ratio of data for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    # Sort by time
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n_samples = len(df_sorted)

    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_df = df_sorted[:train_end]
    val_df = df_sorted[train_end:val_end]
    test_df = df_sorted[val_end:]

    return train_df, val_df, test_df


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets.

    Args:
        y: Target labels.

    Returns:
        Dictionary mapping class labels to weights.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight(
        "balanced", classes=classes, y=y
    )
    
    return dict(zip(classes, weights))


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value to return if denominator is zero.

    Returns:
        Division result or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Loaded configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    from omegaconf import OmegaConf

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        path: Path to save configuration.
    """
    from omegaconf import OmegaConf

    OmegaConf.save(config, path)


class Timer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation"):
        """Initialize timer.

        Args:
            name: Name of the operation being timed.
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and print duration."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} completed in {format_time(duration)}")

    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return None
        end_time = self.end_time or time.time()
        return end_time - self.start_time
