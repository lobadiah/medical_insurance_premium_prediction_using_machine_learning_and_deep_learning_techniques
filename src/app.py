import argparse
import sys
import os
from predict import InsurancePredictor
from utils import setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description='Medical Insurance Premium Predictor CLI')
    parser.add_argument('--age', type=int, required=True, help='Age of the person')
    parser.add_argument('--diabetes', type=int, choices=[0, 1], required=True, help='Diabetes (0: No, 1: Yes)')
    parser.add_argument('--bp', type=int, choices=[0, 1], required=True, help='Blood Pressure Problems (0: No, 1: Yes)')
    parser.add_argument('--transplant', type=int, choices=[0, 1], required=True, help='Any Transplants (0: No, 1: Yes)')
    parser.add_argument('--chronic', type=int, choices=[0, 1], required=True, help='Any Chronic Diseases (0: No, 1: Yes)')
    parser.add_argument('--height', type=float, required=True, help='Height in cm')
    parser.add_argument('--weight', type=float, required=True, help='Weight in kg')
    parser.add_argument('--allergies', type=int, choices=[0, 1], required=True, help='Known Allergies (0: No, 1: Yes)')
    parser.add_argument('--cancer', type=int, choices=[0, 1], required=True, help='History of Cancer in Family (0: No, 1: Yes)')
    parser.add_argument('--surgeries', type=int, required=True, help='Number of Major Surgeries')

    args = parser.parse_args()

    features = [
        args.age, args.diabetes, args.bp, args.transplant, args.chronic,
        args.height, args.weight, args.allergies, args.cancer, args.surgeries
    ]

    try:
        predictor = InsurancePredictor()
        predictions = predictor.predict(features)

        print("\n" + "="*40)
        print("PREDICTION RESULTS")
        print("="*40)
        for model, price in predictions.items():
            print(f"{model:25}: TRY {price:,.2f}")
        print("="*40 + "\n")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
