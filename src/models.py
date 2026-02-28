from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os
import numpy as np

class ModelFactory:
    @staticmethod
    def create_linear_regression():
        return LinearRegression()

    @staticmethod
    def create_random_forest(n_estimators=100, max_depth=10, random_state=42):
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)

    @staticmethod
    def create_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
        return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)

    @staticmethod
    def create_ann(input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

def evaluate_model(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def save_sklearn_model(model, filename, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, filename))

def save_keras_model(model, filename, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, filename))

def load_sklearn_model(filename, models_dir='models'):
    return joblib.load(os.path.join(models_dir, filename))

def load_keras_model(filename, models_dir='models'):
    from tensorflow.keras.models import load_model
    return load_model(os.path.join(models_dir, filename))
