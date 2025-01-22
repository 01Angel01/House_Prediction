import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import default_logger as logger


class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            logger.info(f"Loading configuration from {self.config_path}")

            if not self.config_path.exists():
                logger.error(f"Configuration file not found at {self.config_path}")
                raise FileNotFoundError(
                    f"Configuration file not found at {self.config_path}"
                )

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            logger.info(f"Configuration loaded successfully: {config}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        value = self.config.get(key, default)
        if value is None:
            logger.warning(
                f"Key '{key}' not found in configuration. Using default value: {default}"
            )
        return value


# Create default configuration instance
config = Config()
