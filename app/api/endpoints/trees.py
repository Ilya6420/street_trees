import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from ...services.model_service import TreeHealthModelService
from ...schemas.trees import CSVPredictionResponse, PredictionResponse, TreeInput
from ...utils.constants import ALLOWED_FEATURES


router = APIRouter()
model_service = TreeHealthModelService()


@router.post("/predict", response_model=PredictionResponse)
def predict_tree_health(tree_data: TreeInput):
    """Predict the health status of a single tree."""
    try:
        prediction = model_service.predict(tree_data.model_dump())
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/predict/csv", response_model=CSVPredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """Predict health status for multiple trees from a CSV file."""
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # # Validate required columns
        # required_columns = model_service
        missing_columns = set(ALLOWED_FEATURES) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )

        # Make predictions
        predictions = []
        success_count = 0

        for _, row in df.iterrows():
            try:
                prediction = model_service.predict(TreeInput(**row.to_dict()).model_dump())
                predictions.append({
                    "input": row.to_dict(),
                    "prediction": prediction
                })
                success_count += 1
            except Exception as e:
                predictions.append({
                    "input": row.to_dict(),
                    "error": str(e)
                })

        success_rate = success_count / len(df) if len(df) > 0 else 0

        return {
            "predictions": predictions,
            "total_trees": len(df),
            "success_rate": success_rate
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )
