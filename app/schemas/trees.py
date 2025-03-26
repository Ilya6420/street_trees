from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TreeInput(BaseModel):
    tree_dbh: float = Field(..., description="Tree diameter at breast height")
    curb_loc: Literal["OnCurb", "OffsetFromCurb"] = Field(..., description="Curb location")
    spc_common: str = Field(..., description="Tree species")
    steward: Optional[Literal["1or2", "3or4", "4orMore", "Unknown"]] = Field(..., description="Tree steward")
    guards: Optional[Literal["Helpful", "Harmful", "Unsure", "Unknown"]] = Field(..., description="Tree guards")
    sidewalk: Literal["NoDamage", "Damage"] = Field(..., description="Sidewalk condition")
    user_type: Literal["TreesCount Staff", "Volunteer", "NYC Parks Staff"] = Field(..., description="User type")
    root_stone: Literal["Yes", "No"] = Field(..., description="Root stone")
    root_grate: Literal["Yes", "No"] = Field(..., description="Root grate")
    root_other: Literal["Yes", "No"] = Field(..., description="Root other")
    trunk_wire: Literal["Yes", "No"] = Field(..., description="Trunk wire")
    trnk_light: Literal["Yes", "No"] = Field(..., description="Trunk light")
    trnk_other: Literal["Yes", "No"] = Field(..., description="Trunk other")
    brch_light: Literal["Yes", "No"] = Field(..., description="Branch light")
    brch_shoe: Literal["Yes", "No"] = Field(..., description="Branch shoe")
    brch_other: Literal["Yes", "No"] = Field(..., description="Branch other")
    borough: Literal["Manhattan", "Queens", "Brooklyn", "Staten Island", "Bronx"] = Field(..., description="Borough")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")


class PredictionResponse(BaseModel):
    health_status: str
    confidence: float
    raw_prediction: int


class SingleTreePrediction(BaseModel):
    input: TreeInput
    prediction: PredictionResponse


class CSVPredictionResponse(BaseModel):
    predictions: List[SingleTreePrediction]
    total_trees: int
    success_rate: float


class FeatureImportance(BaseModel):
    Feature: str
    Importance: float
