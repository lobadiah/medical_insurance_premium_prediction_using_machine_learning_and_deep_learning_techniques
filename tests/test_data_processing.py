import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import preprocess_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset
        self.df = pd.DataFrame({
            'Age': [25, 45, 60, 30, 50],
            'Diabetes': [0, 1, 0, 1, 0],
            'BloodPressureProblems': [0, 1, 1, 0, 1],
            'AnyTransplants': [0, 0, 1, 0, 0],
            'AnyChronicDiseases': [0, 1, 0, 0, 1],
            'Height': [175, 170, 165, 180, 160],
            'Weight': [70, 85, 75, 90, 65],
            'KnownAllergies': [0, 1, 0, 0, 1],
            'HistoryOfCancerInFamily': [0, 0, 1, 0, 0],
            'NumberOfMajorSurgeries': [0, 2, 1, 0, 1],
            'PremiumPrice': [15000, 25000, 35000, 20000, 30000]
        })

    def test_preprocess_data_shapes(self):
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(self.df, test_size=0.2)

        # With 5 samples and test_size=0.2, we expect 4 train and 1 test
        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(X_train.shape[1], 10)
        self.assertEqual(len(feature_columns), 10)

    def test_preprocess_data_scaling(self):
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(self.df)

        # Check if mean is close to 0 and std close to 1 for scaled training data
        self.assertTrue(np.allclose(X_train.mean(axis=0), 0, atol=1e-5))
        self.assertTrue(np.allclose(X_train.std(axis=0), 1, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
