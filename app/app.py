from fastapi import FastAPI
import uvicorn

from app.api.endpoints import trees, utils


app = FastAPI(
    title="NYC Tree Health Classifier API",
    description="API for predicting the health status of NYC street trees",
    version="1.0.0"
)

app.include_router(trees.router, prefix="/trees")
app.include_router(utils.router, prefix="/utils")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
