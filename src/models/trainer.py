import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from src.utils.logger import default_logger as logger
from src.models.model import ModelFactory
from src.utils.config import config
import json
import os


class ModelTrainer:
    """Class for training and evaluating regression models"""

    def __init__(self, experiment_name: str = "house_price_prediction"):
        """
        Initialize ModelTrainer

        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models_info = {}
        self.best_model = None
        self.setup_mlflow()
        logger.info(f"Initialized ModelTrainer with experiment: {experiment_name}")

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression evaluation metrics"""
        try:
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking"""
        try:
            tracking_uri = config.get("mlflow").get(
                "tracking_uri", "sqlite:///mlflow.db"
            )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info("MLflow setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train a single regression model"""
        try:
            logger.info(f"Training {model_type} model")

            # Handle missing values
            imputer = SimpleImputer(strategy="mean")
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(
                    X_train, columns=[f"Feature_{i}" for i in range(X_train.shape[1])]
                )
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(
                    X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])]
                )

            X_train = pd.DataFrame(
                imputer.fit_transform(X_train), columns=X_train.columns
            )
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

            # Validate data
            if X_train.empty or X_test.empty:
                raise ValueError("Training or test data is empty. Cannot train model.")

            # Create and train model
            model = ModelFactory.create_model(model_type)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)

            # Log with MLflow
            with mlflow.start_run(run_name=model_type, nested=True) as run:
                mlflow.log_params(
                    model.get_params() if hasattr(model, "get_params") else {}
                )
                mlflow.log_metrics(metrics)

                # Log feature importance if available
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(
                        zip(X_train.columns, model.feature_importances_)
                    )
                    importance_file = "feature_importance.json"
                    with open(importance_file, "w") as f:
                        json.dump(feature_importance, f)
                    mlflow.log_artifact(importance_file)
                    os.remove(importance_file)

                # Log model
                mlflow.sklearn.log_model(
                    model, model_type, registered_model_name=f"house_price_{model_type}"
                )

            # Store model info
            model_info = {
                "model": model,
                "metrics": metrics,
                "run_id": run.info.run_id,
            }
            self.models_info[model_type] = model_info

            logger.info(
                f"Completed training {model_type} model with metrics: {metrics}"
            )
            return model_info
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            raise

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Train all configured regression models"""
        try:
            logger.info("Starting training for all models")
            model_types = ModelFactory.get_model_config().keys()

            best_rmse = float("inf")
            for model_type in model_types:
                logger.info(f"Training model: {model_type}")
                result = self.train_model(model_type, X_train, y_train, X_test, y_test)
                if result["metrics"]["rmse"] < best_rmse:
                    best_rmse = result["metrics"]["rmse"]
                    self.best_model = result["model"]

            logger.info(f"Best model selected with RMSE: {best_rmse}")
        except Exception as e:
            logger.error(f"Error training all models: {str(e)}")
            raise

    def get_best_model(self) -> Dict[str, Any]:
        """Retrieve the best model and its details"""
        try:
            if not self.best_model:
                raise ValueError("No best model has been selected.")
            return next(
                model_info
                for model_info in self.models_info.values()
                if model_info["model"] == self.best_model
            )
        except Exception as e:
            logger.error(f"Error retrieving the best model: {e}")
            raise
