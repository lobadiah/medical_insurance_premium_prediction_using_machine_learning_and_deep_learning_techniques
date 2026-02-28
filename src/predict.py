import os
import pandas as pd
import numpy as np
import joblib
from data_processing import load_preprocessors
from models import load_sklearn_model, load_keras_model
from utils import setup_logging

logger = setup_logging()

class InsurancePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        try:
            self.scaler, self.feature_columns = load_preprocessors(models_dir)
            self.models = {
                'Linear Regression': load_sklearn_model('linear_regression_model.pkl', models_dir),
                'Random Forest': load_sklearn_model('random_forest_model.pkl', models_dir),
                'Gradient Boosting': load_sklearn_model('gradient_boosting_model.pkl', models_dir),
                'Neural Network': load_keras_model('ann_model.keras', models_dir)
            }
        except FileNotFoundError as e:
            logger.error(f"Missing model files: {e}. Please run train_model.py first.")
            raise

    def predict(self, input_data):
        """
        input_data: list or array of features
        """
        df_input = pd.DataFrame([input_data], columns=self.feature_columns)
        input_scaled = self.scaler.transform(df_input)

        predictions = {}
        for name, model in self.models.items():
            try:
                if name == 'Neural Network':
                    pred = model.predict(input_scaled, verbose=0)[0][0]
                else:
                    pred = model.predict(input_scaled)[0]
                predictions[name] = float(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")

        return predictions

if __name__ == "__main__":
    try:
        predictor = InsurancePredictor()
        # Sample prediction
        sample = [25, 0, 0, 0, 0, 175, 70, 0, 0, 0]
        preds = predictor.predict(sample)
        logger.info(f"Predictions for sample {sample}: {preds}")
    except Exception:
        pass
