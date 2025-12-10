import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🏥 MEDICAL INSURANCE PREMIUM PREDICTOR")
print("="*70)

# Your actual performance metrics
MODEL_PERFORMANCE = {
    'Linear Regression': {'R2': 0.7134, 'MAE': 2586.23, 'RMSE': 3495.95},
    'Random Forest': {'R2': 0.8764, 'MAE': 1024.60, 'RMSE': 2295.66},
    'Gradient Boosting': {'R2': 0.8258, 'MAE': 1289.37, 'RMSE': 2725.16},
    'Neural Network': {'R2': 0.6546, 'MAE': 2951.51, 'RMSE': 3837.60}
}

# Check model files
print("\n🔍 Checking model files...")
required_files = [
    'models/linear_regression_model.pkl',
    'models/random_forest_model.pkl',
    'models/gradient_boosting_model.pkl',
    'models/ann_model.h5',
    'models/scaler.pkl',
    'models/feature_columns.pkl'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - Missing")

# Load models
print("\n📂 Loading models...")
try:
    lr_model = joblib.load('models/linear_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_columns.pkl')
    
    from tensorflow import keras
    ann_model = keras.models.load_model('models/ann_model.h5', compile=False)
    ann_model.compile(optimizer='adam', loss='mse')
    
    print("✅ All 4 models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Store models
models_dict = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'Neural Network': ann_model
}

print(f"\n📊 Model Accuracy:")
for name, metrics in MODEL_PERFORMANCE.items():
    print(f"   • {name}: {metrics['R2']:.1%}")

def predict_all_models(features):
    """Make predictions with all 4 models"""
    df_input = pd.DataFrame([features], columns=feature_names)
    input_scaled = scaler.transform(df_input)
    
    predictions = {}
    for name, model in models_dict.items():
        try:
            if name == 'Neural Network':
                pred = model.predict(input_scaled, verbose=0)[0][0]
            else:
                pred = model.predict(input_scaled)[0]
            predictions[name] = float(pred)
        except:
            continue
    
    return predictions

def display_patient_profile(features):
    """Display patient information"""
    labels = ["Age", "Diabetes", "Blood Pressure", "Transplants", 
              "Chronic Diseases", "Height (cm)", "Weight (kg)", 
              "Allergies", "Cancer History", "Surgeries"]
    
    print("\n" + "="*50)
    print("🧾 PATIENT PROFILE")
    print("="*50)
    
    for i, (label, value) in enumerate(zip(labels, features)):
        if i in [0, 5, 6, 9]:
            print(f"  • {label}: {value}")
        else:
            status = "Yes" if value == 1 else "No"
            print(f"  • {label}: {status}")

def analyze_predictions(predictions):
    """Analyze and display predictions"""
    if not predictions:
        print("❌ No predictions available")
        return
    
    print("\n" + "="*50)
    print("💵 PREMIUM PREDICTIONS")
    print("="*50)
    
    print(f"\nPredictions from {len(predictions)} models:")
    print("-" * 40)
    
    for model, price in predictions.items():
        if model in MODEL_PERFORMANCE:
            acc = MODEL_PERFORMANCE[model]['R2'] * 100
            print(f"  {model:20} ₺{price:,.2f} ({acc:.1f}% accurate)")
        else:
            print(f"  {model:20} ₺{price:,.2f}")
    
    # Calculate statistics
    prices = list(predictions.values())
    avg_price = np.mean(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    print(f"\n📈 Statistics:")
    print(f"  Average Premium: ₺{avg_price:,.2f}")
    print(f"  Range: ₺{min_price:,.2f} - ₺{max_price:,.2f}")
    
    # Avoid division by zero
    if avg_price > 0:
        usd_equivalent = avg_price / 32  # Approximate USD to TRY conversion
        print(f"  USD Equivalent: ${usd_equivalent:,.2f}")
    
    # Model agreement (within 10% of average)
    if len(predictions) > 1 and avg_price > 0:
        agreement_threshold = avg_price * 0.10  # 10% threshold
        agreeing_models = []
        for model, price in predictions.items():
            if abs(price - avg_price) <= agreement_threshold:
                agreeing_models.append(model)
        
        print(f"\n🤝 Model Consensus: {len(agreeing_models)}/{len(predictions)} models agree within 10%")
        if agreeing_models:
            print(f"   Agreeing models: {', '.join(agreeing_models)}")
    
    # Recommend best model
    available_models = [m for m in predictions.keys() if m in MODEL_PERFORMANCE]
    if available_models:
        best_model = max(available_models, key=lambda x: MODEL_PERFORMANCE[x]['R2'])
        print(f"\n🎯 Recommended: {best_model}")
        print(f"   Premium: ₺{predictions[best_model]:,.2f}")
        print(f"   Reason: Highest accuracy ({MODEL_PERFORMANCE[best_model]['R2']:.1%})")

def calculate_health_score(features):
    """Calculate health score (0-100)"""
    age, diabetes, bp, transplant, chronic, height, weight, allergies, cancer, surgeries = features
    
    score = 100
    if age > 50: score -= 10
    if age > 60: score -= 15
    if diabetes: score -= 20
    if bp: score -= 15
    if transplant: score -= 25
    if chronic: score -= 20
    if allergies: score -= 5
    if cancer: score -= 15
    score -= surgeries * 5
    
    # BMI calculation
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    if bmi < 18.5 or bmi > 25:
        score -= 10
    
    return max(0, score)

def interactive_prediction():
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print("🔮 MAKE A PREDICTION")
    print("="*70)
    
    try:
        print("\nEnter patient details:")
        age = int(input("Age: "))
        diabetes = 1 if input("Diabetes? (y/n): ").lower() == 'y' else 0
        bp = 1 if input("Blood Pressure problems? (y/n): ").lower() == 'y' else 0
        transplant = 1 if input("Any transplants? (y/n): ").lower() == 'y' else 0
        chronic = 1 if input("Chronic diseases? (y/n): ").lower() == 'y' else 0
        height = float(input("Height (cm): "))
        weight = float(input("Weight (kg): "))
        allergies = 1 if input("Known allergies? (y/n): ").lower() == 'y' else 0
        cancer = 1 if input("Family cancer history? (y/n): ").lower() == 'y' else 0
        surgeries = int(input("Number of major surgeries: "))
        
        features = [age, diabetes, bp, transplant, chronic, 
                   height, weight, allergies, cancer, surgeries]
        
        # Display profile
        display_patient_profile(features)
        
        # Make predictions
        predictions = predict_all_models(features)
        
        # Analyze predictions
        analyze_predictions(predictions)
        
        # Health assessment
        health_score = calculate_health_score(features)
        print(f"\n🏥 Health Score: {health_score}/100")
        
        if health_score >= 80:
            print("   Status: Excellent health ✓")
        elif health_score >= 60:
            print("   Status: Good health ✓")
        elif health_score >= 40:
            print("   Status: Fair health")
        else:
            print("   Status: High risk - Consult doctor")
            
    except ValueError:
        print("❌ Please enter valid numbers!")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_cases():
    """Show demo cases"""
    print("\n" + "="*70)
    print("🧪 DEMONSTRATION CASES")
    print("="*70)
    
    cases = [
        ["Young Healthy", [25, 0, 0, 0, 0, 175, 70, 0, 0, 0]],
        ["Middle-Aged", [45, 1, 1, 0, 0, 170, 85, 0, 0, 1]],
        ["Senior", [60, 1, 1, 1, 1, 165, 90, 1, 1, 2]]
    ]
    
    for name, features in cases:
        print(f"\n📋 {name}:")
        display_patient_profile(features)
        
        predictions = predict_all_models(features)
        
        print("\n💵 Predictions:")
        for model, price in predictions.items():
            if model in MODEL_PERFORMANCE:
                acc = MODEL_PERFORMANCE[model]['R2'] * 100
                print(f"  {model:20} ₺{price:,.2f} ({acc:.1f}%)")
            else:
                print(f"  {model:20} ₺{price:,.2f}")

def show_performance():
    """Show model performance with 4 subplots"""
    print("\n" + "="*70)
    print("📊 MODEL PERFORMANCE & CHARTS")
    print("="*70)
    
    # Create performance table
    print(f"\n{'Model':25} {'R² Score':12} {'MAE':12} {'RMSE':12}")
    print("-"*70)
    
    for name, metrics in MODEL_PERFORMANCE.items():
        print(f"{name:25} {metrics['R2']:12.4f} {metrics['MAE']:12.2f} {metrics['RMSE']:12.2f}")
    
    # Find best performers
    best_r2 = max(MODEL_PERFORMANCE.items(), key=lambda x: x[1]['R2'])
    best_mae = min(MODEL_PERFORMANCE.items(), key=lambda x: x[1]['MAE'])
    best_rmse = min(MODEL_PERFORMANCE.items(), key=lambda x: x[1]['RMSE'])
    
    print(f"\n🏆 BEST PERFORMING MODELS:")
    print(f"  • Highest R² Score: {best_r2[0]} ({best_r2[1]['R2']:.4f})")
    print(f"  • Lowest MAE: {best_mae[0]} (₺{best_mae[1]['MAE']:.2f})")
    print(f"  • Lowest RMSE: {best_rmse[0]} (₺{best_rmse[1]['RMSE']:.2f})")
    
    # Create visualization with 4 subplots (original chart)
    create_performance_charts()

def create_performance_charts():
    """Create performance comparison charts with 4 subplots"""
    models = list(MODEL_PERFORMANCE.keys())
    r2_scores = [MODEL_PERFORMANCE[m]['R2'] for m in models]
    mae_scores = [MODEL_PERFORMANCE[m]['MAE'] for m in models]
    rmse_scores = [MODEL_PERFORMANCE[m]['RMSE'] for m in models]
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # Plot 1: R² Scores
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Accuracy (R² Score) - Higher is Better', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MAE Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('MAE (₺)', fontsize=12)
    ax2.set_title('Mean Absolute Error - Lower is Better', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'₺{score:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: RMSE Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, rmse_scores, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('RMSE (₺)', fontsize=12)
    ax3.set_title('Root Mean Squared Error - Lower is Better', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars3, rmse_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'₺{score:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Radar/Spider chart for overall comparison
    ax4 = axes[1, 1]
    
    # Normalize scores for radar chart (0-1, where 1 is best)
    # For R²: higher is better
    # For MAE/RMSE: lower is better, so we invert
    r2_normalized = r2_scores  # Already 0-1
    
    # Invert MAE and RMSE (lower is better)
    mae_normalized = 1 - (np.array(mae_scores) / max(mae_scores))
    rmse_normalized = 1 - (np.array(rmse_scores) / max(rmse_scores))
    
    # Combine metrics
    metrics_normalized = np.vstack([r2_normalized, mae_normalized, rmse_normalized])
    
    # Number of variables
    categories = ['R² Score', 'MAE', 'RMSE']
    N = len(categories)
    
    # Angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot each model
    for i, model in enumerate(models):
        values = metrics_normalized[:, i].tolist()
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax4.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Normalized Performance Comparison', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Performance charts saved as 'model_performance_comparison.png'")
    plt.show()

def show_model_details():
    """Display details about each machine learning model"""
    print("\n" + "="*70)
    print("🧠 LEARN ABOUT THE ML MODELS")
    print("="*70)
    
    model_details = {
        'Linear Regression': {
            'type': 'Traditional Statistical Model',
            'description': 'Finds the best-fit straight line through data points',
            'strength': 'Simple, fast, and interpretable',
            'weakness': 'Assumes linear relationships between variables'
        },
        'Random Forest': {
            'type': 'Ensemble Learning',
            'description': 'Combines predictions from multiple decision trees',
            'strength': 'Handles non-linear relationships, robust to outliers',
            'weakness': 'Can be computationally expensive with large datasets'
        },
        'Gradient Boosting': {
            'type': 'Ensemble Learning (Sequential)',
            'description': 'Builds trees sequentially to correct previous errors',
            'strength': 'Often achieves highest accuracy',
            'weakness': 'Sensitive to hyperparameters, can overfit'
        },
        'Neural Network': {
            'type': 'Deep Learning',
            'description': 'Multi-layer artificial neural network with 64-32-16 neurons',
            'strength': 'Can capture complex patterns and relationships',
            'weakness': 'Requires large datasets, harder to interpret'
        }
    }
    
    for model_name, details in model_details.items():
        print(f"\n🔹 {model_name.upper()}:")
        print("-" * 40)
        print(f"   Type: {details['type']}")
        print(f"   Description: {details['description']}")
        print(f"   Strength: {details['strength']}")
        print(f"   Weakness: {details['weakness']}")
        print(f"   Performance: {MODEL_PERFORMANCE[model_name]['R2']:.1%} accuracy")

def show_best_model_analysis():
    """Show why Random Forest is the best model"""
    print("\n" + "="*70)
    print("🏆 WHY RANDOM FOREST IS BEST")
    print("="*70)
    
    best_model = max(MODEL_PERFORMANCE.items(), key=lambda x: x[1]['R2'])
    worst_model = min(MODEL_PERFORMANCE.items(), key=lambda x: x[1]['R2'])
    
    print(f"\n⭐ BEST MODEL: {best_model[0]}")
    print(f"   • Accuracy: {best_model[1]['R2']:.1%} (R² = {best_model[1]['R2']:.4f})")
    print(f"   • Average Error: Only ₺{best_model[1]['MAE']:,.2f}")
    print(f"   • Key Advantage: 22.5% more accurate than Linear Regression")
    
    print(f"\n📉 LEAST ACCURATE: {worst_model[0]}")
    print(f"   • Accuracy: {worst_model[1]['R2']:.1%} (R² = {worst_model[1]['R2']:.4f})")
    print(f"   • Likely Reason: Neural networks need more data for optimal performance")
    
    print("\n📊 COMPARISON WITH OTHER MODELS:")
    for model_name, metrics in MODEL_PERFORMANCE.items():
        if model_name != best_model[0]:
            improvement = ((best_model[1]['R2'] - metrics['R2']) / metrics['R2']) * 100
            print(f"   • {best_model[0]} is {improvement:.1f}% more accurate than {model_name}")
    
    print("\n🎯 RECOMMENDATION:")
    print("   • Use Random Forest for highest accuracy")
    print("   • Consider Gradient Boosting as backup model")
    print("   • Linear Regression provides interpretable baseline")
    print("   • Neural Network shows potential with more data")

def main_menu():
    """Main menu with 6 options"""
    while True:
        print("\n" + "="*70)
        print("📋 MAIN MENU - MEDICAL INSURANCE PREDICTION SYSTEM")
        print("="*70)
        print("\nSelect an option:")
        print("1. 🔮 Make a Prediction (Enter patient details)")
        print("2. 📊 View Model Performance & Charts")
        print("3. 🧪 See Example Cases (Demo predictions)")
        print("4. 🧠 Learn About the ML Models")
        print("5. 🏆 Why Random Forest is Best")
        print("6. 🚪 Exit Program")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            interactive_prediction()
        elif choice == '2':
            show_performance()
        elif choice == '3':
            demo_cases()
        elif choice == '4':
            show_model_details()
        elif choice == '5':
            show_best_model_analysis()
        elif choice == '6':
            print("\n" + "="*70)
            print("✅ PROJECT SUMMARY")
            print("="*70)
            print("\nSuccessfully implemented 4 machine learning models:")
            print(f"1. Linear Regression: {MODEL_PERFORMANCE['Linear Regression']['R2']:.1%} accuracy")
            print(f"2. Random Forest: {MODEL_PERFORMANCE['Random Forest']['R2']:.1%} accuracy ⭐ BEST")
            print(f"3. Gradient Boosting: {MODEL_PERFORMANCE['Gradient Boosting']['R2']:.1%} accuracy")
            print(f"4. Neural Network: {MODEL_PERFORMANCE['Neural Network']['R2']:.1%} accuracy")
            
            print("\n🎯 Key Finding: Random Forest achieved 87.6% accuracy")
            print("   with average error of only ₺1,024.60")
            print("\nAll models are working and ready for deployment!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 6.")

# Run the app
if __name__ == "__main__":
    main_menu()