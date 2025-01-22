import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple, Dict, Optional
from pathlib import Path
from src.utils.logger import default_logger as logger
from src.utils.config import config


class DataProcessor:
    """Data preprocessing pipeline for train.csv"""

    def __init__(self, preprocessing_path: Optional[str] = None):
        """
        Initialize data processor

        Args:
            preprocessing_path: Path to save/load preprocessing objects
        """
        self.preprocessing_path = preprocessing_path or config.get("data").get(
            "preprocessing_path", "models/preprocessing"
        )
        self.trained = False
        logger.info("Initialized DataProcessor")

    def _prepare_preprocessing_path(self) -> None:
        """Create preprocessing directory if it doesn't exist"""
        Path(self.preprocessing_path).mkdir(parents=True, exist_ok=True)

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "SalePrice"
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit preprocessors and transform data

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            Tuple of transformed features and target
        """
        try:
            logger.info("Starting fit_transform process")

            # Split features and target
            X = df.drop(columns=[target_col], errors="ignore")
            y = df[target_col] if target_col in df.columns else None

            # Process categorical and numerical columns
            categorical_cols = config.get("categorical_features", [])
            numerical_cols = config.get("numerical_features", [])

            # Define transformers
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            categorical_transformer = Pipeline(
                steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
            )

            # Define preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numerical_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )

            # Fit and transform the data
            logger.info("Fitting preprocessors and transforming data")
            X_processed = self.preprocessor.fit_transform(X)

            self.trained = True
            logger.info("Fit_transform completed successfully")

            return X_processed, y

        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.trained:
            raise ValueError("DataProcessor not fitted. Call fit_transform first.")

        try:
            logger.info("Starting transform process")

            # Transform data
            X_processed = self.preprocessor.transform(df)
            logger.info("Transform completed successfully")
            return X_processed

        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise

    def save_preprocessors(self) -> None:
        """Save preprocessor objects"""
        try:
            logger.info(f"Saving preprocessors to {self.preprocessing_path}")
            self._prepare_preprocessing_path()

            # Save preprocessor
            joblib.dump(
                self.preprocessor, Path(self.preprocessing_path) / "preprocessor.joblib"
            )

            logger.info("Preprocessors saved successfully")

        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise

    def load_preprocessors(self) -> None:
        """Load preprocessor objects"""
        try:
            logger.info(f"Loading preprocessors from {self.preprocessing_path}")

            # Load preprocessor
            preprocessor_path = Path(self.preprocessing_path) / "preprocessor.joblib"
            self.preprocessor = joblib.load(preprocessor_path)

            self.trained = True
            logger.info("Preprocessors loaded successfully")

        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
