from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
from src.utils.logger import default_logger as logger
from src.data.data_preprocessor import DataProcessor
from src.api.schemas import (
    HousePredictionRequest,
    HousePredictionResponse,
    ModelInfo,
    ModelMetrics,
)
import os

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using trained models",
    version="1.0.0",
)

# Global variables
model = None
preprocessor = None
model_info = None


def load_best_model() -> tuple:
    """Load the best model and its associated metadata from MLflow."""
    client = MlflowClient()

    experiment = client.get_experiment_by_name("house_price_experiment")
    if not experiment:
        logger.error("No experiment found with name 'house_price_experiment'")
        raise ValueError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_model_rmse ASC"],
    )
    if not runs:
        logger.error("No runs found in the experiment")
        raise ValueError("No runs found in the experiment")

    best_run = runs[0]
    run_id = best_run.info.run_id
    logger.info(f"Best run ID: {run_id}")

    # Load model
    model_uri = f"runs:/{run_id}/artifacts/decision_tree/model.pkl"
    model = mlflow.sklearn.load_model(model_uri)

    # Load preprocessor
    preprocessor_uri = f"runs:/{run_id}/artifacts/preprocessor"
    preprocessor = mlflow.pyfunc.load_model(preprocessor_uri)

    # Prepare model info
    metrics = best_run.data.metrics
    model_info = ModelInfo(
        model_name="DecisionTreeRegressor",
        model_version=1,
        run_id=run_id,
        metrics=ModelMetrics(
            best_model_mse=metrics.get("best_model_mse", 0.0),
            best_model_rmse=metrics.get("best_model_rmse", 0.0),
            best_model_r2=metrics.get("best_model_r2", 0.0),
        ),
    )

    return model, preprocessor, model_info


@app.on_event("startup")
async def startup_event():
    """Load the best model and preprocessor on startup."""
    global model, preprocessor, model_info
    try:
        logger.info("Loading the best model and preprocessor from MLflow.")
        model, preprocessor, model_info = load_best_model()
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load model on startup.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "House Price Prediction API",
        "model_info": model_info.dict() if model_info else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=HousePredictionResponse)
async def predict(request: HousePredictionRequest):
    """Predict house price based on input features."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded.")
    try:
        data = pd.DataFrame([request.dict()])
        processed_data = preprocessor.transform(data)
        predicted_price = model.predict(processed_data)[0]
        return HousePredictionResponse(
            predicted_price=predicted_price, model_info=model_info
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if model and preprocessor else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_info": model_info.dict() if model_info else None,
    }
