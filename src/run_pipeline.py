import os
import sys
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataProcessor
from src.models.trainer import ModelTrainer
from src.utils.logger import default_logger as logger
from src.utils.config import config

# Define root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = ROOT_DIR / "mlflow.db"


def setup_mlflow():
    """Setup MLflow configuration"""
    try:
        # Set MLflow tracking URI explicitly
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

        # Set experiment
        experiment_name = config.get("mlflow").get(
            "experiment_name", "house_price_experiment"
        )
        try:
            mlflow.create_experiment(experiment_name)
        except:
            pass

        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow setup completed. Tracking URI: {mlflow.get_tracking_uri()}"
        )
        logger.info(f"Experiment name: {experiment_name}")
        return experiment_name

    except Exception as e:
        logger.error(f"Error setting up MLflow: {str(e)}")
        raise


def run_pipeline():
    """Run the complete training pipeline"""
    try:
        logger.info("Starting pipeline execution")

        # Setup MLflow first
        experiment_name = setup_mlflow()

        # Start MLflow run for the entire pipeline
        with mlflow.start_run(run_name="full_pipeline") as parent_run:
            logger.info(f"Started pipeline run with ID: {parent_run.info.run_id}")

            # 1. Load Data
            logger.info("Step 1: Loading data")
            data_loader = DataLoader()
            df = data_loader.load_data()

            if not data_loader.validate_data(df):
                logger.error("Data validation failed. Check required columns.")
                raise ValueError("Data validation failed")

            mlflow.log_param("data_shape", str(df.shape))
            mlflow.log_param("data_columns", str(list(df.columns)))

            # 2. Preprocessing
            logger.info("Step 2: Preprocessing data")
            target_col = config.get("data").get("target_column", "SalePrice")
            preprocessor = DataProcessor()
            X, y = preprocessor.fit_transform(df, target_col=target_col)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])

            preprocessor.save_preprocessors()

            # 3. Model Training
            logger.info("Step 3: Training and evaluating models")
            trainer = ModelTrainer(experiment_name)
            trainer.train_all_models(X_train, y_train, X_test, y_test)

            # Log best model info
            best_model = trainer.get_best_model()
            mlflow.log_param("best_model_type", best_model["model"].__class__.__name__)
            mlflow.log_metrics(
                {f"best_model_{k}": v for k, v in best_model["metrics"].items()}
            )

            logger.info("Pipeline execution completed successfully")
            return True

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset-db", action="store_true", help="Delete existing MLflow database"
    )
    args = parser.parse_args()

    if args.reset_db and MLFLOW_DB.exists():
        MLFLOW_DB.unlink()
        logger.info("Deleted existing MLflow database")

    os.makedirs(ROOT_DIR / "logs", exist_ok=True)
    os.makedirs(ROOT_DIR / "models", exist_ok=True)
    os.makedirs(ROOT_DIR / "artifacts", exist_ok=True)

    run_pipeline()
