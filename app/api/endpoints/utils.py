from fastapi import APIRouter


router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NYC Tree Health Classifier API",
        "version": "1.0.0",
        "description": "API for predicting the health status of NYC street trees",
        "endpoints": {
            "/predict": "POST - Predict tree health status",
            "/predict/csv": "POST - Predict from CSV file",
            "/health": "GET - API health check"
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
