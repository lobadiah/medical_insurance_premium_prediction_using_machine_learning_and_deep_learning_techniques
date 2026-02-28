import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    """Load the dataset from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def preprocess_data(df, target_column='PremiumPrice', test_size=0.2, random_state=42):
    """Clean and split the data into training and testing sets."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def save_preprocessors(scaler, feature_columns, models_dir='models'):
    """Save the scaler and feature columns for later use."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(feature_columns, os.path.join(models_dir, 'feature_columns.pkl'))

def load_preprocessors(models_dir='models'):
    """Load the scaler and feature columns."""
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    return scaler, feature_columns
