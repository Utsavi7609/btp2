import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv("clean_dataset.csv")

def evaluate(model_prefix):

    v_true = df["self_reported_valence"]
    a_true = df["self_reported_arousal"]

    v_pred = df[f"inferred_valence_{model_prefix}"]
    a_pred = df[f"inferred_arousal_{model_prefix}"]

    mae_v = mean_absolute_error(v_true, v_pred)
    mae_a = mean_absolute_error(a_true, a_pred)

    corr_v = np.corrcoef(v_true, v_pred)[0,1]
    corr_a = np.corrcoef(a_true, a_pred)[0,1]

    acc_v = np.mean(np.abs(v_true - v_pred) <= 1)
    acc_a = np.mean(np.abs(a_true - a_pred) <= 1)

    return mae_v, mae_a, corr_v, corr_a, acc_v, acc_a


models = ["llama", "mistral", "llama4"]

for m in models:

    results = evaluate(m)

    print(f"\nModel: {m}")
    print(f"MAE Valence: {results[0]:.3f}")
    print(f"MAE Arousal: {results[1]:.3f}")
    print(f"Corr Valence: {results[2]:.3f}")
    print(f"Corr Arousal: {results[3]:.3f}")
    print(f"Acc Valence (±1): {results[4]:.3f}")
    print(f"Acc Arousal (±1): {results[5]:.3f}")