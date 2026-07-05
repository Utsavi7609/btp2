import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

print("="*60)
print("FINAL BTP PERFORMANCE EVALUATION (Zero-Shot vs Few-Shot)")
print("="*60)

DATA_DIR = "InHouseCollectedData"

def calculate_metrics(df, true_v_col, true_a_col, inf_v_col, inf_a_col):
    valid = df.dropna(subset=[true_v_col, true_a_col, inf_v_col, inf_a_col])
    n = len(valid)
    if n == 0:
        return None
    
    tv = valid[true_v_col].astype(float)
    ta = valid[true_a_col].astype(float)
    pv = valid[inf_v_col].astype(float)
    pa = valid[inf_a_col].astype(float)
    
    mae_v = np.mean(np.abs(tv - pv))
    mae_a = np.mean(np.abs(ta - pa))
    
    try:
        r_v, _ = pearsonr(tv, pv)
    except:
        r_v = np.nan
        
    try:
        r_a, _ = pearsonr(ta, pa)
    except:
        r_a = np.nan
        
    acc_v = np.mean(np.abs(tv - pv) <= 1) * 100
    acc_a = np.mean(np.abs(ta - pa) <= 1) * 100
    
    return {
        'N': n,
        'V_MAE': mae_v,
        'A_MAE': mae_a,
        'V_R': r_v,
        'A_R': r_a,
        'V_Acc': acc_v,
        'A_Acc': acc_a
    }

results = []

# Model Files
model_configs = [
    ('Llama-3.3-70B', 'llm_results_llama.csv', 'llm_results_llama_fewshot.csv'),
    ('Qwen3-32B', 'llm_results_mistral.csv', 'llm_results_mistral_fewshot.csv'),
    ('Llama-4-Scout', 'llm_results_llama4.csv', 'llm_results_llama4_fewshot.csv'),
]

for model_name, zs_file, fs_file in model_configs:
    zs_path = os.path.join(DATA_DIR, zs_file)
    fs_path = os.path.join(DATA_DIR, fs_file)
    
    # ─── Zero-Shot Metrics ───
    if os.path.exists(zs_path):
        df_zs = pd.read_csv(zs_path)
        m = calculate_metrics(df_zs, 'self_reported_valence', 'self_reported_arousal', 'inferred_valence', 'inferred_arousal')
        if m:
            results.append({
                'Model': model_name,
                'Method': 'Zero-Shot',
                'Valence MAE': f"{m['V_MAE']:.3f}",
                'Arousal MAE': f"{m['A_MAE']:.3f}",
                'Valence R': f"{m['V_R']:.3f}",
                'Arousal R': f"{m['A_R']:.3f}",
                'Valence Acc ±1': f"{m['V_Acc']:.1f}%",
                'Arousal Acc ±1': f"{m['A_Acc']:.1f}%"
            })
            
    # ─── Few-Shot Metrics ───
    if os.path.exists(fs_path):
        df_fs = pd.read_csv(fs_path)
        m = calculate_metrics(df_fs, 'self_reported_valence', 'self_reported_arousal', 'inferred_valence', 'inferred_arousal')
        if m:
            results.append({
                'Model': model_name,
                'Method': 'Few-Shot (3-Shot)',
                'Valence MAE': f"{m['V_MAE']:.3f}",
                'Arousal MAE': f"{m['A_MAE']:.3f}",
                'Valence R': f"{m['V_R']:.3f}",
                'Arousal R': f"{m['A_R']:.3f}",
                'Valence Acc ±1': f"{m['V_Acc']:.1f}%",
                'Arousal Acc ±1': f"{m['A_Acc']:.1f}%"
            })

df_final = pd.DataFrame(results)
print(df_final.to_string(index=False))

df_final.to_csv('btp_final_comparison.csv', index=False)
print("\n✅ Results saved to btp_final_comparison.csv")
