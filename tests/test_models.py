import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ModelFactory, evaluate_model

class TestModels(unittest.TestCase):
    def test_linear_regression_creation(self):
        model = ModelFactory.create_linear_regression()
        self.assertIsNotNone(model)

    def test_random_forest_creation(self):
        model = ModelFactory.create_random_forest()
        self.assertIsNotNone(model)

    def test_evaluate_model(self):
        y_true = np.array([1000, 2000, 3000])
        y_pred = np.array([1100, 1900, 3100])

        metrics = evaluate_model(y_true, y_pred)

        self.assertIn('R2', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertEqual(metrics['MAE'], 100.0)

if __name__ == '__main__':
    unittest.main()
