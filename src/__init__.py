"""Network Intrusion Detection System - Main Package."""

__version__ = "0.1.0"
__author__ = "Security Research Team"
__email__ = "research@example.com"

from .utils.config import Config
from .utils.utils import setup_logging, set_random_seeds, get_device

__all__ = [
    "Config",
    "setup_logging", 
    "set_random_seeds",
    "get_device",
]
