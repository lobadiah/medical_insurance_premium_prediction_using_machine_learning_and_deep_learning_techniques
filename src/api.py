from flask import Flask, request, jsonify
from predict import InsurancePredictor
from utils import setup_logging
import os

logger = setup_logging()
app = Flask(__name__)

# Initialize predictor
try:
    predictor = InsurancePredictor()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input. Expected "features" key.'}), 400

    try:
        features = data['features']
        if len(features) != len(predictor.feature_columns):
            return jsonify({'error': f'Expected {len(predictor.feature_columns)} features, got {len(features)}'}), 400

        predictions = predictor.predict(features)
        return jsonify({
            'predictions': predictions,
            'units': 'TRY'
        })
    except Exception as e:
        logger.error(f"API prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
