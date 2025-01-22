import pandas as pd
from typing import Optional, Tuple
from src.utils.logger import default_logger as logger
from src.utils.config import config


class DataLoader:
    """Data loading utilities"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader

        Args:
            data_path: Optional path to data file
        """
        self.data_path = data_path or config.get("data_path")
        logger.info(f"Initialized DataLoader with path: {self.data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info("Start loading data...")
            if not self.data_path:
                raise ValueError("Data path is not specified.")
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error("The provided file is empty.")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate loaded data

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if validation passes
        """
        try:
            logger.info("Validating data...")

            # Check required columns
            required_columns = config.get("data").get("required_columns", [])
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                logger.warning(
                    f"Found null values in the following columns:\n{null_counts[null_counts > 0]}"
                )

            logger.info("Data validation completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and target

        Args:
            df: Input DataFrame

        Returns:
            Tuple containing features DataFrame and target Series
        """
        try:
            logger.info("Splitting features and target...")
            target_column = config.get("data").get("target_column", "SalePrice")

            if target_column not in df.columns:
                raise KeyError(f"Target column '{target_column}' not found in data.")

            X = df.drop(columns=[target_column])
            y = df[target_column]

            logger.info(
                f"Split completed. Features shape: {X.shape}, Target shape: {y.shape}"
            )
            return X, y
        except KeyError as e:
            logger.error(f"Error splitting features and target: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during splitting: {e}")
            raise
