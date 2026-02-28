import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess_data, save_preprocessors
from models import ModelFactory, evaluate_model, save_sklearn_model, save_keras_model
from utils import setup_logging, plot_correlation_matrix, plot_distribution

# Setup logging
logger = setup_logging()

def train():
    # Paths
    data_path = os.path.join('data', 'Medicalpremium.csv')
    models_dir = 'models'
    plots_dir = 'plots'

    # Load data
    logger.info(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info(f"Dataset shape: {df.shape}")

    # Visualizations
    logger.info("Generating EDA plots...")
    plot_distribution(df['PremiumPrice'], 'Distribution of Premium Prices',
                      'Premium Price', 'Frequency',
                      os.path.join(plots_dir, 'premium_distribution.png'))
    plot_correlation_matrix(df, os.path.join(plots_dir, 'correlation_matrix.png'))

    # Preprocess
    logger.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)
    save_preprocessors(scaler, feature_columns, models_dir)

    results = {}

    # 1. Linear Regression
    logger.info("Training Linear Regression...")
    lr_model = ModelFactory.create_linear_regression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    results['Linear Regression'] = evaluate_model(y_test, y_pred_lr)
    save_sklearn_model(lr_model, 'linear_regression_model.pkl', models_dir)

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf_model = ModelFactory.create_random_forest()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf)
    save_sklearn_model(rf_model, 'random_forest_model.pkl', models_dir)

    # 3. Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb_model = ModelFactory.create_gradient_boosting()
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    results['Gradient Boosting'] = evaluate_model(y_test, y_pred_gb)
    save_sklearn_model(gb_model, 'gradient_boosting_model.pkl', models_dir)

    # 4. ANN
    logger.info("Training Artificial Neural Network...")
    ann_model = ModelFactory.create_ann(X_train.shape[1])
    ann_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    y_pred_ann = ann_model.predict(X_test).flatten()
    results['Neural Network'] = evaluate_model(y_test, y_pred_ann)
    save_keras_model(ann_model, 'ann_model.keras', models_dir)

    # Model Comparison
    comparison_df = pd.DataFrame(results).T
    logger.info("\nModel Comparison Summary:\n" + comparison_df.round(4).to_string())

    # SHAP Analysis for Random Forest
    logger.info("Computing SHAP values for Random Forest...")
    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        np.save(os.path.join(models_dir, 'shap_values_rf.npy'), shap_values)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'shap_summary_rf.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("SHAP summary plot saved.")
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")

    logger.info("Training completed successfully.")

if __name__ == "__main__":
    train()
