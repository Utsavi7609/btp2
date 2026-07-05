"""
Baseline ML Comparison
Validates LLM predictions against traditional ML
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

df = pd.read_csv("inferred_emotions_llm.csv")

# Features
features = ["hr_mean", "hr_std", "hr_min", "hr_max", "hrv_rmssd"]
X = df[features]

# Targets
y_valence = df["inferred_valence"]
y_arousal = df["inferred_arousal"]

# Split data
X_train, X_test, yv_train, yv_test = train_test_split(
    X, y_valence, test_size=0.2, random_state=42
)
_, _, ya_train, ya_test = train_test_split(
    X, y_arousal, test_size=0.2, random_state=42
)

# Train models
print("Training Random Forest baselines...")
model_v = RandomForestRegressor(n_estimators=100, random_state=42)
model_a = RandomForestRegressor(n_estimators=100, random_state=42)

model_v.fit(X_train, yv_train)
model_a.fit(X_train, ya_train)

# Predictions
pred_v = model_v.predict(X_test)
pred_a = model_a.predict(X_test)

# Evaluation
print("\n" + "="*60)
print("BASELINE MODEL PERFORMANCE")
print("="*60)
print(f"\nValence:")
print(f"  MAE: {mean_absolute_error(yv_test, pred_v):.3f}")
print(f"  R²:  {r2_score(yv_test, pred_v):.3f}")

print(f"\nArousal:")
print(f"  MAE: {mean_absolute_error(ya_test, pred_a):.3f}")
print(f"  R²:  {r2_score(ya_test, pred_a):.3f}")

# Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Valence)")
print("="*60)
for feat, imp in zip(features, model_v.feature_importances_):
    print(f"  {feat:15s}: {imp:.3f}")

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Arousal)")
print("="*60)
for feat, imp in zip(features, model_a.feature_importances_):
    print(f"  {feat:15s}: {imp:.3f}")

print("\n✅ Baseline validation complete")