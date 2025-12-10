import joblib
import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns

def load_models():
    """Load all trained models and components"""
    try:
        print("Loading models and components...")
                # Load models
        lr_model = joblib.load("models/linear_regression_model.pkl")
        rf_model = joblib.load("models/random_forest_model.pkl")
        gb_model = joblib.load("models/gradient_boosting_model.pkl")
        ann_model = joblib.load("models/ann_model.h5")
        # Load preprocessing components
        scaler = joblib.load("models/scaler.pkl")
        feature_columns = joblib.load("models/feature_columns.pkl")
        
        print("✅ All models loaded successfully!")
        return {
            'lr': lr_model,
            'rf': rf_model,
            'gb': gb_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'ann_model': ann_model
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  IMPORTANT: You need to run train_model.py FIRST!")
        print("Run this command: python train_model.py")
        return None

def predict_premium(input_data, models_dict):
    """
    Make predictions using all four models
    
    input_data should be a list in this order:
    [Age, Diabetes, BloodPressureProblems, AnyTransplants, 
     AnyChronicDiseases, Height, Weight, KnownAllergies, 
     HistoryOfCancerInFamily, NumberOfMajorSurgeries]
    """
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data], columns=models_dict['feature_columns'])
    
    # Scale the features
    input_scaled = models_dict['scaler'].transform(df_input)
    
    # Make predictions with all models
    predictions = {
        'Linear Regression': models_dict['lr'].predict(input_scaled)[0],
        'Random Forest': models_dict['rf'].predict(input_scaled)[0],
        'Gradient Boosting': models_dict['gb'].predict(input_scaled)[0],
        'Artificial Neural Network': models_dict['ann_model'].predict(input_scaled)[0][0]
    }
    
    return predictions

def display_patient_profile(input_data):
    """Display patient information in readable format"""
    print("\n" + "="*50)
    print("PATIENT PROFILE")
    print("="*50)
    
    labels = [
        "Age",
        "Diabetes (0=No, 1=Yes)",
        "Blood Pressure Problems (0=No, 1=Yes)",
        "Any Transplants (0=No, 1=Yes)",
        "Any Chronic Diseases (0=No, 1=Yes)",
        "Height (cm)",
        "Weight (kg)",
        "Known Allergies (0=No, 1=Yes)",
        "History of Cancer in Family (0=No, 1=Yes)",
        "Number of Major Surgeries"
    ]
    
    for label, value in zip(labels, input_data):
        print(f"  {label}: {value}")

def predict_from_console():
    """Interactive prediction from console input"""
    
    # Load models first
    models_dict = load_models()
    if models_dict is None:
        return
    
    print("\n" + "="*50)
    print("MEDICAL INSURANCE PREMIUM PREDICTOR")
    print("="*50)
    print("\nPlease enter patient details:\n")
    
    input_data = []
    
    try:
        # Age
        age = int(input("Age (18-100): "))
        while age < 18 or age > 100:
            print("Age must be between 18 and 100")
            age = int(input("Age (18-100): "))
        input_data.append(age)
        
        # Diabetes
        diabetes = input("Diabetes? (yes/no): ").lower()
        while diabetes not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            diabetes = input("Diabetes? (yes/no): ").lower()
        input_data.append(1 if diabetes in ['yes', 'y', '1'] else 0)
        
        # Blood Pressure Problems
        bp = input("Blood Pressure Problems? (yes/no): ").lower()
        while bp not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            bp = input("Blood Pressure Problems? (yes/no): ").lower()
        input_data.append(1 if bp in ['yes', 'y', '1'] else 0)
        
        # Any Transplants
        transplant = input("Any Transplants? (yes/no): ").lower()
        while transplant not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            transplant = input("Any Transplants? (yes/no): ").lower()
        input_data.append(1 if transplant in ['yes', 'y', '1'] else 0)
        
        # Any Chronic Diseases
        chronic = input("Any Chronic Diseases? (yes/no): ").lower()
        while chronic not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            chronic = input("Any Chronic Diseases? (yes/no): ").lower()
        input_data.append(1 if chronic in ['yes', 'y', '1'] else 0)
        
        # Height
        height = float(input("Height (in cm, e.g., 175): "))
        while height < 100 or height > 250:
            print("Height should be between 100cm and 250cm")
            height = float(input("Height (in cm): "))
        input_data.append(height)
        
        # Weight
        weight = float(input("Weight (in kg, e.g., 70): "))
        while weight < 30 or weight > 200:
            print("Weight should be between 30kg and 200kg")
            weight = float(input("Weight (in kg): "))
        input_data.append(weight)
        
        # Known Allergies
        allergies = input("Known Allergies? (yes/no): ").lower()
        while allergies not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            allergies = input("Known Allergies? (yes/no): ").lower()
        input_data.append(1 if allergies in ['yes', 'y', '1'] else 0)
        
        # History of Cancer in Family
        cancer = input("History of Cancer in Family? (yes/no): ").lower()
        while cancer not in ['yes', 'no', 'y', 'n', '1', '0']:
            print("Please enter 'yes' or 'no'")
            cancer = input("History of Cancer in Family? (yes/no): ").lower()
        input_data.append(1 if cancer in ['yes', 'y', '1'] else 0)
        
        # Number of Major Surgeries
        surgeries = int(input("Number of Major Surgeries (0-10): "))
        while surgeries < 0 or surgeries > 10:
            print("Number of surgeries should be between 0 and 10")
            surgeries = int(input("Number of Major Surgeries (0-10): "))
        input_data.append(surgeries)
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return
    
    # Display patient profile
    display_patient_profile(input_data)
    
    # Make predictions
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    
    predictions = predict_premium(input_data, models_dict)
    
    print("\n📊 Predicted Premiums:")
    for model_name, premium in predictions.items():
        print(f"  {model_name}: ₺{premium:,.2f}")
    
    # Calculate average prediction
    avg_premium = np.mean(list(predictions.values()))
    print(f"\n🎯 Average Prediction: ₺{avg_premium:,.2f}")
    print(f"   USD Equivalent: ${avg_premium/83:,.2f}")
    
    # Show which model predicted highest/lowest
    max_model = max(predictions, key=predictions.get)
    min_model = min(predictions, key=predictions.get)
    print(f"\n📈 Highest prediction: {max_model} (₺{predictions[max_model]:,.2f})")
    print(f"📉 Lowest prediction: {min_model} (₺{predictions[min_model]:,.2f})")
    print(f"   Difference: ₺{predictions[max_model] - predictions[min_model]:,.2f}")

def run_test_predictions():
    """Run test predictions with sample data"""
    
    models_dict = load_models()
    if models_dict is None:
        return
    
    print("\n" + "="*50)
    print("TEST PREDICTIONS")
    print("="*50)
    
    # Test cases
    test_cases = [
        # Test 1: Young healthy person
        [25, 0, 0, 0, 0, 175, 70, 0, 0, 0],
        # Test 2: Middle-aged with health issues
        [45, 1, 1, 0, 1, 170, 85, 1, 0, 2],
        # Test 3: Older person with transplants
        [60, 0, 1, 1, 0, 165, 75, 0, 1, 1]
    ]
    
    test_descriptions = [
        "Young healthy person (25yo, no issues)",
        "Middle-aged with health issues (45yo, diabetes, BP problems)",
        "Older person with transplant history (60yo, transplant, cancer family history)"
    ]
    
    for i, (test_data, description) in enumerate(zip(test_cases, test_descriptions)):
        print(f"\n🧪 Test Case {i+1}: {description}")
        print("-" * 40)
        
        predictions = predict_premium(test_data, models_dict)
        
        for model_name, premium in predictions.items():
            print(f"  {model_name}: ₺{premium:,.2f}")
        
        avg_premium = np.mean(list(predictions.values()))
        print(f"  Average: ₺{avg_premium:,.2f}")

def check_model_files():
    """Check if model files exist"""
    required_files = [
        'models/linear_regression_model.pkl',
        'models/random_forest_model.pkl',
        'models/gradient_boosting_model.pkl',
        'models/scaler.pkl',
        'models/feature_columns.pkl',
        'models/ann_model.h5'
    ]
    
    print("Checking for model files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    return missing_files

if __name__ == "__main__":
    print("Medical Insurance Premium Prediction System")
    print("="*50)
    
    # Check if model files exist
    missing_files = check_model_files()
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} model file(s)!")
        print("You need to train the models first.")
        print("Run: python train_model.py")
        
        response = input("\nDo you want to run train_model.py now? (yes/no): ").lower()
        if response == 'yes':
            print("\nRunning train_model.py...")
            os.system("python train_model.py")
        else:
            print("\nPlease run train_model.py first, then run predict.py again.")
            sys.exit(1)
    else:
        print("\n✅ All model files found!")
        
        # Run test predictions
        run_test_predictions()
        
        # Ask for interactive mode
        print("\n" + "="*50)
        response = input("\nDo you want to enter custom patient data? (yes/no): ").lower()
        if response == 'yes':
            predict_from_console()
        
        print("\n" + "="*50)
        print("✅ Prediction system completed!")
        print("="*50)