# FastAPI Tree Classification Service

This repository contains a Python-based application and ML pipeline, providing functionality for data preprocessing, model training, and an example web service interface (if applicable). It follows best practices for structuring a data science or machine learning project.

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Running the Service](#running-the-service)
6. [Linting & Pre-commit Hooks](#linting--pre-commit-hooks)
7. [Docker Usage](#docker-usage)
8. [Notebooks & Training Utilities](#notebooks--training-utilities)
9. [Further Improvements](#further-improvements)

---

# Overview
- **Programming Language**: Python 3.12
- **Application Framework**: FastAPI

# Dependencies
Below is a breakdown of libraries, pinned to specific versions, organized by how they’re used in this project.
## Application deps
Listed in `requirements.txt`:
```
pandas==2.2.3
torch==2.6.0
scikit-learn==1.6.1
fastapi==0.115.11
uvicorn==0.34.0
pydantic==2.10.6
joblib==1.4.2
python-dotenv==1.1.0
python-multipart==0.0.20
```
**Usage**: These are the primary libraries required to run the application (data manipulation, ML model inference, FastAPI, etc.).
## Development deps
Listed in `requirements-dev.txt`:
```
flake8==7.1.1
pip-audit==2.7.3
pre-commit==4.0.1
pytest==8.3.4
```
**Usage**: Tools for code quality (flake8), security auditing (pip-audit), pre-commit hooks, and testing (pytest).
## Notebook deps
Listed in `requirements-notebook.txt`:
```
torch==2.6.0
numpy==2.1.0
scikit-learn==1.6.1
matplotlib==3.10.1
seaborn==0.13.2
folium==0.19.5
joblib==1.4.2
```
**Usage**: These packages are mainly for data analysis, visualization, and some repeated ML functionality in Jupyter notebooks (beyond the application).

---
# Project Structure
```
.
├── Dockerfile                                  # Docker image configuration
├── docker-compose.yml                          # Compose config file
├── README.MD                                   # Documentation
├── requirements-dev.txt                        # Development/test dependencies
├── requirements-notebook.txt                   # EDA/Training dependencies
├── requirements.txt                            # Application  dependencies
├── .flake8                                     # Flake8 configuration
├── .pre-commit-config.yml                      # Pre-commit hooks configuration
├── app                                         # Application Folder
│   ├── api
│   │   └── endpoints
│   │       ├── trees.py            
│   │       └── utils.py
│   ├── core
│   │   ├── model_architecture.py
│   │   └── preprocessing_architecture.py
│   ├── app.py
│   ├── schemas
│   │   └── trees.py
│   ├── services
│   │   └── model_service.py
│   └── utils
│       └── constants.py
└── tests                                        # Folder for future test files
```
## Key Directories:
```app/```: Core source files:

```api/endpoints/```: Defines REST endpoints.

```core/```: Contains model definitions & data preprocessing pipeline schemas.

```schemas/```: Pydantic schemas for validation.

```services/```: Business logic for loading models, making predictions, etc.

```utils/```: Constants.

```app.py```: Main entry point for the service.

```notebooks/```: Jupyter notebooks for data analysis, feature engineering, and model training.

```artifacts/, models/```: Directories for storing generated artifacts and trained models.

```tests/```: Contains automated tests.
# Installation
1. **Clone Repository**
    ```
    git clone <YOUR_REPO_URL>.git
    cd <YOUR_REPO_NAME>
    ```
2. **Install miniconda**

Link: https://www.anaconda.com/docs/getting-started/miniconda/install

3. **Create & activate a virtual environment**
    ```
    conda create -n <env_name> python=3.12
    conda activate <env_name>
    ```
4. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
    If you are developing or running tests, also install development dependencies:
    ```
    pip install -r requirements-dev.txt
    ```
    For EDA and notebook, istall notebook dependencies
    ```
    pip install -r requirements-notebook.txt
    ```
5. **Create kernel for Notebooks**
    ```
    pip install ipykernel
    python -m ipykernel install --user --name=<kernel name>
    ```
6. **Dataset enabling**
    * Create in the root folder ```data``` with ```data/raw``` and ```data/processed``` subfolers.
    * Install dataset from https://www.kaggle.com/datasets/new-york-city/ny-2015-street-tree-census-tree-data/data and put .csv into ```data/raw```
    * Run ```3_data_preprocessing.ipynb``` notebook to get test data for endpoint (optional)
    * In case you want to reproduce model result, run:
        1. ```3_data_preprocessig.ipynb```
        2. ```4_feature_engineering_training.ipynb ```

# Running the Service
1. Start the FastAPI app with Uvicorn:
    ```
    uvicorn app.app:app --host 0.0.0.0 --port 8000
    ```
2. Access the service:
    * Open http://127.0.0.1:8000/docs to see the automatically generated FastAPI docs (Swagger UI).
    * Or http://127.0.0.1:8000/redoc for ReDoc documentation.
3. Endpoints
    * ```GET /```: Root endpoint. Returns basic API information.
    * ```GET /health```: Health check endpoint. Returns JSON indicating service status.
    * ```POST /predict```: Predict the health status for a single tree.

    Example body:
    
    ```json
    {
    "tree_dbh": 10.0,
    "curb_loc": "OnCurb",
    "spc_common": "Oak",
    "steward": "1or2",
    "guards": "Helpful",
    "sidewalk": "NoDamage",
    "user_type": "TreesCount Staff",
    "root_stone": "Yes",
    "root_grate": "No",
    "root_other": "No",
    "trunk_wire": "No",
    "trnk_light": "No",
    "trnk_other": "No",
    "brch_light": "No",
    "brch_shoe": "No",
    "brch_other": "No",
    "borough": "Manhattan",
    "latitude": 40.7128,
    "longitude": -74.0060
    }
    ```
    * ```POST /predict/csv``` – Predict health status in bulk by uploading a CSV file.
        * Example: upload a CSV with columns matching the required features (e.g., tree_dbh, curb_loc, etc.), and the API will return predictions for each row.

# Linting & Pre-commit Hooks
## Flake 8
The project uses Flake8 to check code style:
```
flake8
```
## Pip-audit
The project uses pip-audit to check the dependencies for known vulnerabilites.
## Pre-commit
Pre-commit hooks (e.g., running Flake8 before commits) are configured in ```pre-commit-config```.yaml. To enable pre-commit:
1. Install pre-commit if not already installed:
```
pip install pre-commit
```
2. Run:
```
pre-commit install
```
3. Now, Flake8 and pip-audit checks will run automatically before each commit.

# Docker 
1. Build the Docker Image
```
docker build -t tree-classifier-service .
```
2. Run the Docker Container:
```
docker run -p 8000:8000 -e MODEL_PATH="" -e PREPROCESSOR_PATH="" -e TARGET_ENCODER_PATH="" tree-classifier-service
```

3. Docker-compose

You can also use Docker Compose or other container orchestration if needed.
```
docker compose up --build
```
# Notebooks & Training Utilities
* ```1_eda_general.ipynb```: Performs initial exploratory data analysis.
* ```2_eda_target.ipynb```: Focuses on the EDA around target.
* ```3_data_preprocessing.ipynb```: Covers data cleaning and feature transformations.
* ```4_feature_engineering_training.ipynb```: Further feature engineering and training.

**In notebooks/training_utils/**:
* ```torch_dataset.py```: Custom dataset class for PyTorch.
* ```trainer.py```: General-purpose training/inference functions.
* ```models.py```: DL model architecture.
* ```losses.py```: Custom loss function.
* ```visualization.py```: Plotting tools for model performance.

# Further Improvements
## DS
* More feature Engineering
* Model parameter tuning
* Try tree-based models
## Dev
* add tests
* improve exceptions
* add logging
* add tracing -> jaeger
## MLOPS
* Integrate Mlflow for better experiment tracking
* Dataset Managemet using DVC with External Storage
* Model Management with External Storage