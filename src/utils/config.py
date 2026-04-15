"""Configuration management for Network Intrusion Detection System."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the NIDS system."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file.

        Returns:
            Loaded configuration as DictConfig.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        return OmegaConf.load(self.config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        return OmegaConf.select(self._config, key, default=default)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates.
        """
        OmegaConf.set(self._config, updates)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        OmegaConf.save(self._config, save_path)

    @property
    def data_config(self) -> DictConfig:
        """Get data configuration section."""
        return self._config.data

    @property
    def model_config(self) -> DictConfig:
        """Get model configuration section."""
        return self._config.models

    @property
    def training_config(self) -> DictConfig:
        """Get training configuration section."""
        return self._config.training

    @property
    def evaluation_config(self) -> DictConfig:
        """Get evaluation configuration section."""
        return self._config.evaluation

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of configuration."""
        self._config[key] = value

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path})"
