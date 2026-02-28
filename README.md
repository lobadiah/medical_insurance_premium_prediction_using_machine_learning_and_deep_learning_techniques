# Medical Insurance Premium Prediction System

A professional machine learning project to predict medical insurance premiums based on patient profiles.

## Project Structure

```
├── data/                   # Dataset (CSV)
├── models/                 # Saved models and preprocessors
├── plots/                  # Visualizations and EDA
├── notebook/               # Jupyter Notebooks for exploration
├── src/                    # Source code
│   ├── api.py              # Flask REST API
│   ├── app.py              # Command-line Interface (CLI)
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── models.py           # Model definitions and factory
│   ├── predict.py          # Prediction logic
│   ├── train_model.py      # Training script
│   └── utils.py            # Utility functions (logging, plotting)
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, Gradient Boosting, and Artificial Neural Network (ANN).
- **Explainability**: Integrated SHAP analysis for model interpretability.
- **Dual Interface**: Structured CLI and RESTful API for predictions.
- **Robustness**: Professional logging, error handling, and unit testing.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train all models and generate visualizations:
```bash
python src/train_model.py
```

### Prediction (CLI)

Use the command-line interface to get predictions:
```bash
python src/app.py --age 25 --diabetes 0 --bp 0 --transplant 0 --chronic 0 --height 175 --weight 70 --allergies 0 --cancer 0 --surgeries 0
```

### API

Start the REST API server:
```bash
python src/api.py
```

Predict via POST request:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [25, 0, 0, 0, 0, 175, 70, 0, 0, 0]}'
```

## Testing

Run unit tests:
```bash
python -m unittest discover tests
```

## Dataset

The dataset contains features such as age, health conditions, height, weight, and surgery history to predict the insurance premium price.
