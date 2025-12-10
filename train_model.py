"""
Enhanced train_model.py with SHAP explainability
KEEPING YOUR EXACT STRUCTURE AND CODE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# NEW: Import SHAP
import shap
import warnings
warnings.filterwarnings('ignore')

# Create directories (same as yours)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load YOUR dataset (same as yours)
print("Loading Medicalpremium.csv...")
df = pd.read_csv(r"C:\Users\Mr. Louis Obadiah\Desktop\new_project\Medicalpremium.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values (same as yours)
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics (same as yours)
print("\nDataset statistics:")
print(df.describe())

# Check target distribution (same as yours)
print(f"\nTarget 'PremiumPrice' statistics:")
print(f"Min: {df['PremiumPrice'].min()}")
print(f"Max: {df['PremiumPrice'].max()}")
print(f"Mean: {df['PremiumPrice'].mean():.2f}")
print(f"Std: {df['PremiumPrice'].std():.2f}")

# Visualize premium distribution (same as yours)
plt.figure(figsize=(10, 6))
plt.hist(df['PremiumPrice'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Premium Price')
plt.ylabel('Frequency')
plt.title('Distribution of Premium Prices')
plt.grid(True, alpha=0.3)
plt.savefig('plots/premium_distribution.png')
plt.close()

# Check correlation (same as yours)
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

print("\n📊 Correlation with PremiumPrice:")
correlation_with_target = df.corr()['PremiumPrice'].sort_values(ascending=False)
print(correlation_with_target)

# Prepare features and target (same as yours)
X = df.drop('PremiumPrice', axis=1)
y = df['PremiumPrice']

# Split data (same as yours)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📈 Data Split:")
print(f"Training set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# Scale features (same as yours)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler (same as yours)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

# Initialize results dictionary (same as yours)
results = {}

# ==================== MODEL 1: LINEAR REGRESSION ====================
# EXACTLY THE SAME AS YOUR CODE
print("\n" + "="*50)
print("MODEL 1: LINEAR REGRESSION")
print("="*50)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# Calculate metrics (same as yours)
lr_metrics = {
    'R2': r2_score(y_test, y_pred_lr),
    'MAE': mean_absolute_error(y_test, y_pred_lr),
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
}

results['Linear Regression'] = lr_metrics

print(f"R² Score: {lr_metrics['R2']:.4f}")
print(f"MAE: {lr_metrics['MAE']:.2f}")
print(f"MSE: {lr_metrics['MSE']:.2f}")
print(f"RMSE: {lr_metrics['RMSE']:.2f}")

# Save model (same as yours)
joblib.dump(lr_model, 'models/linear_regression_model.pkl')
print("✅ Linear Regression model saved!")

# ==================== MODEL 2: RANDOM FOREST ====================
# EXACTLY THE SAME AS YOUR CODE
print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST REGRESSOR")
print("="*50)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# Calculate metrics (same as yours)
rf_metrics = {
    'R2': r2_score(y_test, y_pred_rf),
    'MAE': mean_absolute_error(y_test, y_pred_rf),
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
}

results['Random Forest'] = rf_metrics

print(f"R² Score: {rf_metrics['R2']:.4f}")
print(f"MAE: {rf_metrics['MAE']:.2f}")
print(f"MSE: {rf_metrics['MSE']:.2f}")
print(f"RMSE: {rf_metrics['RMSE']:.2f}")

# Feature importance (same as yours)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance (Random Forest):")
print(feature_importance)

# Plot feature importance (same as yours)
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Save model (same as yours)
joblib.dump(rf_model, 'models/random_forest_model.pkl')
print("✅ Random Forest model saved!")

# ==================== NEW: SHAP ANALYSIS FOR RANDOM FOREST ====================
print("\n" + "="*50)
print("SHAP ANALYSIS FOR RANDOM FOREST")
print("="*50)

try:
    print("Computing SHAP values...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Save SHAP values
    np.save('models/shap_values_rf.npy', shap_values)
    print("✅ SHAP values saved!")
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title('SHAP Summary Plot - Random Forest')
    plt.tight_layout()
    plt.savefig('plots/shap_summary_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved!")
    
    # Create SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig('plots/shap_bar_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP bar plot saved!")
    
    # Create dependence plot for top feature
    top_feature = feature_importance.iloc[0]['feature']
    top_feature_idx = list(X.columns).index(top_feature)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values, X_test_scaled, 
                         feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot - {top_feature}')
    plt.tight_layout()
    plt.savefig('plots/shap_dependence_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP dependence plot saved!")
    
    # Print SHAP insights
    print(f"\n🔍 SHAP Insights:")
    print(f"• Base value (average prediction): {explainer.expected_value:.2f}")
    
    # Get mean absolute SHAP values for feature importance
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 5 features by SHAP importance:")
    for i, row in shap_importance.head().iterrows():
        print(f"  {row['feature']}: {row['shap_importance']:.4f}")
        
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {e}")
    print("Continuing with other models...")

# ==================== MODEL 3: GRADIENT BOOSTING (Alternative to ANN) ====================
# EXACTLY THE SAME AS YOUR CODE
print("\n" + "="*50)
print("MODEL 3: GRADIENT BOOSTING REGRESSOR")
print("="*50)

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test_scaled)

# Calculate metrics (same as yours)
gb_metrics = {
    'R2': r2_score(y_test, y_pred_gb),
    'MAE': mean_absolute_error(y_test, y_pred_gb),
    'MSE': mean_squared_error(y_test, y_pred_gb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gb))
}

results['Gradient Boosting'] = gb_metrics

print(f"R² Score: {gb_metrics['R2']:.4f}")
print(f"MAE: {gb_metrics['MAE']:.2f}")
print(f"MSE: {gb_metrics['MSE']:.2f}")
print(f"RMSE: {gb_metrics['RMSE']:.2f}")

# Save model (same as yours)
joblib.dump(gb_model, 'models/gradient_boosting_model.pkl')
print("✅ Gradient Boosting model saved!")

# ==================== MODEL 4: ARTIFICIAL NEURAL NETWORK (ANN) ====================
# EXACTLY THE SAME AS YOUR CODE
print("\n" + "="*50)
print("MODEL 4: ARTIFICIAL NEURAL NETWORK")
print("="*50)

ann_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  
])

ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train
history = ann_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Predict and evaluate
y_pred_ann = ann_model.predict(X_test_scaled).flatten()

ann_metrics = {
    'R2': r2_score(y_test, y_pred_ann),
    'MAE': mean_absolute_error(y_test, y_pred_ann),
    'MSE': mean_squared_error(y_test, y_pred_ann),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ann))
}

results['Neural Network'] = ann_metrics

print(f"R² Score: {ann_metrics['R2']:.4f}")
print(f"MAE: {ann_metrics['MAE']:.2f}")
print(f"MSE: {ann_metrics['MSE']:.2f}")
print(f"RMSE: {ann_metrics['RMSE']:.2f}")

# Save model (same as yours)
ann_model.save('models/ann_model.keras')
print("✅ Neural Network model saved!")

# ==================== MODEL COMPARISON ====================
# EXACTLY THE SAME AS YOUR CODE
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

comparison_df = pd.DataFrame(results).T
print("\n📊 Performance Comparison:")
print(comparison_df.round(4))

# Visual comparison (same as yours)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# R² Score comparison
axes[0, 0].bar(comparison_df.index, comparison_df['R2'], color=['blue', 'green', 'orange', 'red'])
axes[0, 0].set_title('R² Score Comparison')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(comparison_df['R2']):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

# RMSE comparison
axes[0, 1].bar(comparison_df.index, comparison_df['RMSE'], color=['blue', 'green', 'orange', 'red'])
axes[0, 1].set_title('RMSE Comparison')
axes[0, 1].set_ylabel('RMSE')
for i, v in enumerate(comparison_df['RMSE']):
    axes[0, 1].text(i, v + 50, f'{v:.0f}', ha='center')

# MAE comparison
axes[1, 0].bar(comparison_df.index, comparison_df['MAE'], color=['blue', 'green', 'orange', 'red'])
axes[1, 0].set_title('MAE Comparison')
axes[1, 0].set_ylabel('MAE')
for i, v in enumerate(comparison_df['MAE']):
    axes[1, 0].text(i, v + 50, f'{v:.0f}', ha='center')

# Actual vs Predicted for Random Forest (best model)
axes[1, 1].scatter(y_test, y_pred_rf, alpha=0.5)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual Premium')
axes[1, 1].set_ylabel('Predicted Premium')
axes[1, 1].set_title('Actual vs Predicted (Random Forest)')

plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.close()

print("✅ Model comparison plot saved!")

# ==================== NEW: SAMPLE PREDICTIONS WITH SHAP EXPLANATIONS ====================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS WITH SHAP EXPLANATIONS")
print("="*60)

# Create some test samples (same as yours)
sample_data = [
    # Young healthy person
    [25, 0, 0, 0, 0, 175, 70, 0, 0, 0],
    # Middle-aged with health issues
    [45, 1, 1, 0, 1, 170, 85, 1, 0, 2],
    # Older person with transplants
    [60, 0, 1, 1, 0, 165, 75, 0, 1, 1]
]

sample_df = pd.DataFrame(sample_data, columns=X.columns)

# Scale the samples (same as yours)
sample_scaled = scaler.transform(sample_df)

# Make predictions with all models (same as yours)
for i, sample in enumerate(sample_data):
    print(f"\nSample {i+1}:")
    print(f"  Age: {sample[0]}, Diabetes: {sample[1]}, BP Problems: {sample[2]}")
    print(f"  Transplant: {sample[3]}, Chronic Disease: {sample[4]}")
    print(f"  Height: {sample[5]}cm, Weight: {sample[6]}kg")
    print(f"  Allergies: {sample[7]}, Cancer History: {sample[8]}, Surgeries: {sample[9]}")
    
    pred_lr = lr_model.predict(sample_scaled[i:i+1])[0]
    pred_rf = rf_model.predict(sample_scaled[i:i+1])[0]
    pred_gb = gb_model.predict(sample_scaled[i:i+1])[0]
    pred_ann = ann_model.predict(sample_scaled[i:i+1]).flatten()[0]
    
    print(f"  LR Prediction: ₺{pred_lr:,.2f}")
    print(f"  RF Prediction: ₺{pred_rf:,.2f}")
    print(f"  GB Prediction: ₺{pred_gb:,.2f}")
    print(f"  ANN Prediction: ₺{pred_ann:,.2f}")
    
    # ==================== NEW: SHAP ANALYSIS FOR RANDOM FOREST ====================
print("\n" + "="*50)
print("SHAP ANALYSIS FOR RANDOM FOREST")
print("="*50)

try:
    print("Computing SHAP values...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Save SHAP values
    np.save('models/shap_values_rf.npy', shap_values)
    print("✅ SHAP values saved!")
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    plt.title('SHAP Summary Plot - Random Forest')
    plt.tight_layout()
    plt.savefig('plots/shap_summary_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved!")
    
    # Create SHAP bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig('plots/shap_bar_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP bar plot saved!")
    
    # Create dependence plot for top feature
    top_feature = feature_importance.iloc[0]['feature']
    top_feature_idx = list(X.columns).index(top_feature)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature_idx, shap_values, X_test_scaled, 
                         feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot - {top_feature}')
    plt.tight_layout()
    plt.savefig('plots/shap_dependence_rf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP dependence plot saved!")
    
    # Print SHAP insights - FIXED THE FORMATTING ERROR HERE
    print(f"\n🔍 SHAP Insights:")
    
    # Get the expected value - handle different return types
    expected_value = explainer.expected_value
    
    # Check if expected_value is an array or scalar
    if isinstance(expected_value, np.ndarray):
        if expected_value.size == 1:
            expected_val = expected_value.item()
            print(f"• Base value (average prediction): {expected_val:.2f}")
        else:
            print(f"• Base value shape: {expected_value.shape}")
            print(f"• First base value: {expected_value[0]:.2f}")
    else:
        # It's a scalar
        print(f"• Base value (average prediction): {expected_value:.2f}")
    
    # Get mean absolute SHAP values for feature importance
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 5 features by SHAP importance:")
    for i, row in shap_importance.head().iterrows():
        print(f"  {row['feature']}: {row['shap_importance']:.4f}")
        
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {str(e)}")
    print("Continuing with other models...")

# ==================== NEW: SAMPLE PREDICTIONS WITH SHAP EXPLANATIONS ====================
print("\n" + "="*60)
print("SAMPLE PREDICTIONS WITH SHAP EXPLANATIONS")
print("="*60)

# Make predictions with all models (same as yours)
for i, sample in enumerate(sample_data):
    print(f"\nSample {i+1}:")
    print(f"  Age: {sample[0]}, Diabetes: {sample[1]}, BP Problems: {sample[2]}")
    print(f"  Transplant: {sample[3]}, Chronic Disease: {sample[4]}")
    print(f"  Height: {sample[5]}cm, Weight: {sample[6]}kg")
    print(f"  Allergies: {sample[7]}, Cancer History: {sample[8]}, Surgeries: {sample[9]}")
    
    pred_lr = lr_model.predict(sample_scaled[i:i+1])[0]
    pred_rf = rf_model.predict(sample_scaled[i:i+1])[0]
    pred_gb = gb_model.predict(sample_scaled[i:i+1])[0]
    pred_ann = ann_model.predict(sample_scaled[i:i+1]).flatten()[0]
    
    print(f"  LR Prediction: ₺{pred_lr:,.2f}")
    print(f"  RF Prediction: ₺{pred_rf:,.2f}")
    print(f"  GB Prediction: ₺{pred_gb:,.2f}")
    print(f"  ANN Prediction: ₺{pred_ann:,.2f}")
    
    # NEW: SHAP explanation for Random Forest prediction
    try:
        # Get SHAP values for this sample
        explainer = shap.TreeExplainer(rf_model)
        shap_values_sample = explainer.shap_values(sample_scaled[i:i+1])[0]
        
        print(f"\n  🔍 SHAP Explanation for Random Forest Prediction:")
        
        # Handle expected value properly
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            if expected_value.size == 1:
                base_value = expected_value.item()
            else:
                base_value = expected_value[0]
        else:
            base_value = expected_value
            
        print(f"    Base value (average premium): ₺{base_value:.2f}")
        
        # Create DataFrame for feature contributions
        contributions = pd.DataFrame({
            'Feature': X.columns,
            'Value': sample,
            'SHAP_Effect': shap_values_sample
        })
        
        # Sort by absolute SHAP value
        contributions['Abs_SHAP'] = np.abs(contributions['SHAP_Effect'])
        contributions = contributions.sort_values('Abs_SHAP', ascending=False)
        
        # Show top 3 contributing features
        print(f"    Top contributing features:")
        for _, row in contributions.head(3).iterrows():
            direction = "increases" if row['SHAP_Effect'] > 0 else "decreases"
            print(f"      • {row['Feature']} = {row['Value']}: {direction} premium by ₺{abs(row['SHAP_Effect']):.2f}")
        
        print(f"    Sum of SHAP effects: ₺{sum(shap_values_sample):.2f}")
        print(f"    Final prediction: ₺{pred_rf:,.2f}")
        
    except Exception as e:
        print(f"  ⚠️ Could not generate SHAP explanation: {str(e)}")

# ==================== NEW: CREATE SHAP REPORT ====================
print("\n" + "="*60)
print("GENERATING SHAP INSIGHTS REPORT")
print("="*60)

try:
    # Load SHAP values if they exist
    if os.path.exists('models/shap_values_rf.npy'):
        shap_values = np.load('models/shap_values_rf.npy')
        
        # Create a comprehensive SHAP report
        print("\n📋 SHAP INSIGHTS REPORT:")
        print("-" * 40)
        
        # 1. Global feature importance
        shap_importance = pd.DataFrame({
            'Feature': X.columns,
            'Mean_|SHAP|': np.abs(shap_values).mean(0),
            'Mean_SHAP': shap_values.mean(0)
        }).sort_values('Mean_|SHAP|', ascending=False)
        
        print("\n1. GLOBAL FEATURE IMPORTANCE (Based on SHAP):")
        for i, row in shap_importance.head(5).iterrows():
            effect = "increases" if row['Mean_SHAP'] > 0 else "decreases"
            print(f"   {row['Feature']}: {effect} premium by ₺{abs(row['Mean_SHAP']):.2f} on average")
        
        # 2. Feature interactions
        print("\n2. KEY INSIGHTS:")
        
        # Find features with strongest positive/negative impact
        pos_impact = shap_importance[shap_importance['Mean_SHAP'] > 0].head(3)
        neg_impact = shap_importance[shap_importance['Mean_SHAP'] < 0].head(3)
        
        if len(pos_impact) > 0:
            print(f"   Features that INCREASE premium the most:")
            for _, row in pos_impact.iterrows():
                print(f"   • {row['Feature']}: +₺{row['Mean_SHAP']:.2f}")
        
        if len(neg_impact) > 0:
            print(f"\n   Features that DECREASE premium the most:")
            for _, row in neg_impact.iterrows():
                print(f"   • {row['Feature']}: -₺{abs(row['Mean_SHAP']):.2f}")
        
        # 3. Model interpretability
        print("\n3. MODEL INTERPRETABILITY:")
        
        # Get explainer again for expected value
        explainer = shap.TreeExplainer(rf_model)
        expected_value = explainer.expected_value
        
        # Handle different expected_value types
        if isinstance(expected_value, np.ndarray):
            if expected_value.size == 1:
                base_val = expected_value.item()
            else:
                base_val = expected_value[0]
        else:
            base_val = expected_value
            
        print(f"   • The model's average prediction (base value) is: ₺{base_val:.2f}")
        print(f"   • For individual predictions, SHAP shows how each feature moves the prediction from the base value")
        print(f"   • Positive SHAP values increase the premium prediction")
        print(f"   • Negative SHAP values decrease the premium prediction")
        
        print("\n✅ SHAP report generated successfully!")
        
except Exception as e:
    print(f"⚠️ Could not generate SHAP report: {str(e)}")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)

# Find best model (same as yours, but with more detail)
best_model_name = comparison_df['R2'].idxmax()
best_model_metrics = comparison_df.loc[best_model_name]

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   R² Score: {best_model_metrics['R2']:.4f}")
print(f"   RMSE: {best_model_metrics['RMSE']:.2f}")
print(f"   MAE: {best_model_metrics['MAE']:.2f}")

print("\n📁 Created Files:")
print("-" * 40)
print("📂 models/")
print("   ├── linear_regression_model.pkl")
print("   ├── random_forest_model.pkl")
print("   ├── gradient_boosting_model.pkl")
print("   ├── ann_model.keras")
print("   ├── scaler.pkl")
print("   ├── feature_columns.pkl")
print("   └── shap_values_rf.npy")

print("\n📂 plots/")
print("   ├── premium_distribution.png")
print("   ├── correlation_matrix.png")
print("   ├── feature_importance.png")
print("   ├── model_comparison.png")
print("   ├── shap_summary_rf.png")
print("   ├── shap_bar_rf.png ")
print("   └── shap_dependence_rf.png")

print("\n🎯 What SHAP adds to your project:")
print("-" * 40)
print("1. EXPLAINABILITY: Understand WHY the model makes certain predictions")
print("2. FEATURE IMPORTANCE: More accurate than traditional feature_importance")
print("3. TRANSPARENCY: See how each feature affects each prediction")
print("4. TRUST: Build confidence in your model's decisions")
print("5. INSIGHTS: Discover patterns and relationships in your data")

print("\n🔍 To view SHAP explanations:")
print("1. Check the 'plots/' folder for SHAP visualizations")
print("2. Look at the sample predictions above for individual explanations")
print("3. Use the SHAP values to understand feature impacts")

print("\n🚀 Next Steps:")
print("1. Run: python predict.py (for individual predictions)")
print("2. Check the new SHAP plots in the 'plots/' folder")
print("3. Use SHAP insights to improve your model or business decisions")

print("\n" + "="*60)
print("📊 PROJECT ENHANCED WITH EXPLAINABLE AI (XAI)!")
print("="*60)