"""
Comprehensive Validation - LLM vs ML Baselines
Includes MAE, Pearson Correlation, R², Feature Importance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

print("="*70)
print("COMPREHENSIVE VALIDATION: LLM vs ML Baselines")
print("="*70)

# Load data
df = pd.read_csv("inferred_emotions_llm.csv")

# Remove any NaN values
df = df.dropna(subset=["hr_mean", "hr_std", "hr_min", "hr_max", "hrv_rmssd"])

print(f"\nDataset: {len(df)} samples")
print(f"Users: {df['user_id'].nunique()}")

# Features
features = ["hr_mean", "hr_std", "hr_min", "hr_max", "hrv_rmssd"]
X = df[features]

y_valence = df["inferred_valence"]
y_arousal = df["inferred_arousal"]

# Split data
X_train, X_test, yv_train, yv_test = train_test_split(
    X, y_valence, test_size=0.2, random_state=42
)
_, _, ya_train, ya_test = train_test_split(
    X, y_arousal, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVM": SVR(kernel='rbf'),
    "MLP": MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, random_state=42)
}

results = []

print("\n" + "="*70)
print("VALENCE PREDICTION")
print("="*70)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, yv_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(yv_test, pred)
    r2 = r2_score(yv_test, pred)
    corr, p_value = pearsonr(yv_test, pred)
    
    results.append({
        'Model': name,
        'Target': 'Valence',
        'MAE': mae,
        'R²': r2,
        'Pearson r': corr,
        'p-value': p_value
    })
    
    print(f"  MAE: {mae:.3f}")
    print(f"  R²:  {r2:.3f}")
    print(f"  Pearson r: {corr:.3f} (p={p_value:.4f})")

print("\n" + "="*70)
print("AROUSAL PREDICTION")
print("="*70)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    model.fit(X_train, ya_train)
    pred = model.predict(X_test)
    
    mae = mean_absolute_error(ya_test, pred)
    r2 = r2_score(ya_test, pred)
    corr, p_value = pearsonr(ya_test, pred)
    
    results.append({
        'Model': name,
        'Target': 'Arousal',
        'MAE': mae,
        'R²': r2,
        'Pearson r': corr,
        'p-value': p_value
    })
    
    print(f"  MAE: {mae:.3f}")
    print(f"  R²:  {r2:.3f}")
    print(f"  Pearson r: {corr:.3f} (p={p_value:.4f})")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("validation_results.csv", index=False)

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(results_df.to_string(index=False))

# Feature Importance (Random Forest)
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*70)

rf_v = RandomForestRegressor(n_estimators=100, random_state=42)
rf_a = RandomForestRegressor(n_estimators=100, random_state=42)

rf_v.fit(X_train, yv_train)
rf_a.fit(X_train, ya_train)

print("\nValence:")
for feat, imp in sorted(zip(features, rf_v.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat:15s}: {imp:.3f}")

print("\nArousal:")
for feat, imp in sorted(zip(features, rf_a.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat:15s}: {imp:.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Valence comparison
valence_results = results_df[results_df['Target'] == 'Valence']
axes[0].barh(valence_results['Model'], valence_results['MAE'])
axes[0].set_xlabel('Mean Absolute Error')
axes[0].set_title('Valence Prediction - Model Comparison')
axes[0].invert_yaxis()

# Arousal comparison
arousal_results = results_df[results_df['Target'] == 'Arousal']
axes[1].barh(arousal_results['Model'], arousal_results['MAE'])
axes[1].set_xlabel('Mean Absolute Error')
axes[1].set_title('Arousal Prediction - Model Comparison')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("baseline_comparison.png", dpi=300)
print("\n✅ Saved: baseline_comparison.png")
plt.show()

print("\n" + "="*70)
print("✅ VALIDATION COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - validation_results.csv")
print("  - baseline_comparison.png")