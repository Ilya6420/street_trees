import os

from dotenv import load_dotenv
load_dotenv()
import joblib
import pandas as pd
import torch

from ..core.model_architecture import TreeClassifier


class TreeHealthModelService:
    def __init__(self,
                 model_path=os.getenv('MODEL_PATH'),
                 preprocessor_pipe_path=os.getenv('PREPROCESSOR_PATH'),
                 target_encoder_path=os.getenv('TARGET_ENCODER_PATH')):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            self.model = self._load_model(model_path)
            self.preprocessor_pipe = self._load_preprocessor(preprocessor_pipe_path)
            self.target_encoder = self._load_target_encoder(target_encoder_path)
        except Exception as e:
            print(f"Error loading model components: {str(e)}")
            raise

        # Load label encodings
        self.health_mapping = {
            0: "Poor",
            1: "Fair",
            2: "Good",
        }

    def _load_model(self, model_path):
        """Load the trained model."""
        try:
            model = TreeClassifier(input_size=27, num_classes=3).to(self.device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def _load_preprocessor(self, preprocessor_path):
        """Load the preprocessor pipeline."""
        try:
            return joblib.load(preprocessor_path)
        except Exception as e:
            raise RuntimeError(f"Error loading preprocessor: {str(e)}")

    def _load_target_encoder(self, encoder_path):
        """Load the target encoder."""
        try:
            return joblib.load(encoder_path)
        except Exception as e:
            raise RuntimeError(f"Error loading target encoder: {str(e)}")

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        df_transformed = self.preprocessor_pipe.transform(df)
        return torch.FloatTensor(df_transformed.values).to(self.device)

    def predict(self, input_data):
        """Make prediction for input data."""
        try:
            # Preprocess input
            input_tensor = self.preprocess_input(input_data)
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                prediction = predicted.item()

            # Convert prediction to human-readable format
            health_status = self.health_mapping[prediction]

            return {
                "health_status": health_status,
                "confidence": float(torch.softmax(outputs, dim=1).max().item()),
                "raw_prediction": int(prediction)
            }

        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
