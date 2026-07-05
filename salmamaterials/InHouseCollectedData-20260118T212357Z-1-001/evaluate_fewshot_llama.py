import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("="*60)
print("COMPARING ZERO-SHOT VS FEW-SHOT ON K-EMOCON (Llama-3.3-70B)")
print("="*60)

# Load zero-shot metrics
try:
    df_zero = pd.read_csv('kemocon_validation_metrics.csv')
    llama_zero = df_zero[df_zero['model'] == 'Llama-3.3-70B'].iloc[0]
    z_mae_v = llama_zero['mae_valence']
    z_mae_a = llama_zero['mae_arousal']
    z_r_v = llama_zero['pearson_r_valence']
    z_r_a = llama_zero['pearson_r_arousal']
except:
    z_mae_v, z_mae_a, z_r_v, z_r_a = 'N/A', 'N/A', 'N/A', 'N/A'

# Load few-shot predictions
df_few = pd.read_csv('kemocon_llama_results.csv')

# Clean
# Drop rows where inference failed
df_few['inferred_valence'] = pd.to_numeric(df_few['inferred_valence'], errors='coerce')
df_few['inferred_arousal'] = pd.to_numeric(df_few['inferred_arousal'], errors='coerce')
valid = df_few.dropna(subset=['true_valence', 'true_arousal', 'inferred_valence', 'inferred_arousal'])

if len(valid) > 0:
    tv = valid['true_valence'].values
    ta = valid['true_arousal'].values
    pv = valid['inferred_valence'].values
    pa = valid['inferred_arousal'].values
    
    f_mae_v = np.mean(np.abs(tv - pv))
    f_mae_a = np.mean(np.abs(ta - pa))
    
    try: f_r_v, _ = pearsonr(tv, pv)
    except: f_r_v = np.nan
    try: f_r_a, _ = pearsonr(ta, pa)
    except: f_r_a = np.nan

    f_acc_v = np.mean(np.abs(tv - pv) <= 1) * 100
    f_acc_a = np.mean(np.abs(ta - pa) <= 1) * 100

    print(f"Number of valid samples: {len(valid)}/28\n")
    
    print(f"--- VALENCE ---")
    print(f"Zero-Shot MAE: {z_mae_v}  |  Few-Shot MAE: {f_mae_v:.3f}")
    print(f"Zero-Shot r:   {z_r_v}  |  Few-Shot r:   {f_r_v:.3f}")
    
    print(f"\n--- AROUSAL ---")
    print(f"Zero-Shot MAE: {z_mae_a}  |  Few-Shot MAE: {f_mae_a:.3f}")
    print(f"Zero-Shot r:   {z_r_a}  |  Few-Shot r:   {f_r_a:.3f}")
    
    print(f"\nFew-Shot Acc ±1 -> Valence: {f_acc_v:.1f}% | Arousal: {f_acc_a:.1f}%")
else:
    print("Could not evaluate Few-Shot. Valid array length is 0.")
