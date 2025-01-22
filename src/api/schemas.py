from pydantic import BaseModel, Field
from typing import Dict, Optional


class ModelMetrics(BaseModel):
    """Schema for model metrics"""

    best_model_mse: float = Field(
        ..., description="Mean Squared Error of the best model"
    )
    best_model_rmse: float = Field(
        ..., description="Root Mean Squared Error of the best model"
    )
    best_model_r2: float = Field(..., description="R-squared score of the best model")


class ModelInfo(BaseModel):
    """Schema for detailed model information"""

    model_name: str = Field(..., description="Name of the model")
    model_version: int = Field(..., description="Version of the registered model")
    run_id: str = Field(..., description="Run ID of the model in MLflow")
    metrics: ModelMetrics = Field(..., description="Metrics of the best model")


class HousePredictionRequest(BaseModel):
    """Request schema for house price prediction"""

    MSSubClass: int = Field(
        ..., description="Identifies the type of dwelling involved in the sale"
    )
    MSZoning: str = Field(
        ..., description="Identifies the general zoning classification of the sale"
    )
    LotFrontage: Optional[float] = Field(
        None, description="Linear feet of street connected to property"
    )
    LotArea: float = Field(..., description="Lot size in square feet")
    Street: str = Field(..., description="Type of road access to property")
    Alley: Optional[str] = Field(None, description="Type of alley access to property")
    LotShape: str = Field(..., description="General shape of property")
    LandContour: str = Field(..., description="Flatness of the property")
    Utilities: str = Field(..., description="Type of utilities available")
    LotConfig: str = Field(..., description="Lot configuration")
    LandSlope: str = Field(..., description="Slope of property")
    Neighborhood: str = Field(
        ..., description="Physical locations within Ames city limits"
    )
    Condition1: str = Field(..., description="Proximity to main road or railroad")
    Condition2: str = Field(
        ..., description="Proximity to main road or railroad (if a second is present)"
    )
    BldgType: str = Field(..., description="Type of dwelling")
    HouseStyle: str = Field(..., description="Style of dwelling")
    OverallQual: int = Field(
        ..., description="Rates the overall material and finish of the house"
    )
    OverallCond: int = Field(
        ..., description="Rates the overall condition of the house"
    )
    YearBuilt: int = Field(..., description="Original construction date")
    YearRemodAdd: int = Field(
        ...,
        description="Remodel date (same as construction date if no remodeling or additions)",
    )
    RoofStyle: str = Field(..., description="Type of roof")
    RoofMatl: str = Field(..., description="Roof material")
    Exterior1st: str = Field(..., description="Exterior covering on house")
    Exterior2nd: str = Field(
        ..., description="Exterior covering on house (if more than one material)"
    )
    MasVnrArea: Optional[float] = Field(
        None, description="Masonry veneer area in square feet"
    )
    ExterQual: str = Field(
        ..., description="Evaluates the quality of the material on the exterior"
    )
    ExterCond: str = Field(
        ...,
        description="Evaluates the present condition of the material on the exterior",
    )
    Foundation: str = Field(..., description="Type of foundation")
    BsmtQual: Optional[str] = Field(
        None, description="Evaluates the height of the basement"
    )
    BsmtCond: Optional[str] = Field(
        None, description="Evaluates the general condition of the basement"
    )
    BsmtExposure: Optional[str] = Field(
        None, description="Refers to walkout or garden level walls"
    )
    TotalBsmtSF: float = Field(..., description="Total square feet of basement area")

    class Config:
        schema_extra = {
            "example": {
                "MSSubClass": 60,
                "MSZoning": "RL",
                "LotFrontage": 65.0,
                "LotArea": 8450.0,
                "Street": "Pave",
                "Alley": None,
                "LotShape": "Reg",
                "LandContour": "Lvl",
                "Utilities": "AllPub",
                "LotConfig": "Inside",
                "LandSlope": "Gtl",
                "Neighborhood": "CollgCr",
                "Condition1": "Norm",
                "Condition2": "Norm",
                "BldgType": "1Fam",
                "HouseStyle": "2Story",
                "OverallQual": 7,
                "OverallCond": 5,
                "YearBuilt": 2003,
                "YearRemodAdd": 2003,
                "RoofStyle": "Gable",
                "RoofMatl": "CompShg",
                "Exterior1st": "VinylSd",
                "Exterior2nd": "VinylSd",
                "MasVnrArea": 196.0,
                "ExterQual": "Gd",
                "ExterCond": "TA",
                "Foundation": "PConc",
                "BsmtQual": "Gd",
                "BsmtCond": "TA",
                "BsmtExposure": "No",
                "TotalBsmtSF": 856.0,
            }
        }


class HousePredictionResponse(BaseModel):
    """Response schema for house price prediction"""

    predicted_price: float = Field(..., description="Predicted price of the house")
    model_info: ModelInfo = Field(..., description="Information about the best model")
