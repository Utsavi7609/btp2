"""
generate_final_results.py
=========================
Combines the in-house dataset metrics and K-EmoCon validation metrics
into a single, clean final_validation_table.csv.

Follows user-provided actual model names:
- "llama" -> Llama-3.3-70B-Versatile
- "mistral" -> Qwen/Qwen3-32B
- "llama4" -> Meta-Llama/Llama-4-Scout-17B-16E-Instruct
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

print("="*60)
print("GENERATING FINAL VALIDATION TABLE")
print("="*60)

# Paths
in_house_data = Path("InHouseCollectedData/clean_dataset.csv")
kemocon_metrics_file = Path("kemocon_validation_metrics.csv")

final_records = []

# ─── 1. Re-calculate In-House Data Metrics ───────────────────────
if in_house_data.exists():
    df_in = pd.read_csv(in_house_data)
    
    # Ensure numeric
    df_in['self_reported_valence'] = pd.to_numeric(df_in['self_reported_valence'], errors='coerce')
    df_in['self_reported_arousal'] = pd.to_numeric(df_in['self_reported_arousal'], errors='coerce')
    
    # Model mappings
    models = {
        'llama': {
            'v_col': 'inferred_valence_llama_first',
            'a_col': 'inferred_arousal_llama_first',
            'name': 'Llama-3.3-70B-Versatile'
        },
        'mistral': {
            'v_col': 'inferred_valence_mistral',
            'a_col': 'inferred_arousal_mistral',
            'name': 'Qwen3-32B'
        },
        'llama4': {
            'v_col': 'inferred_valence_llama4',
            'a_col': 'inferred_arousal_llama4',
            'name': 'Llama-4-Scout-17B'
        }
    }
    
    for key, info in models.items():
        if info['v_col'] in df_in.columns and info['a_col'] in df_in.columns:
            valid = df_in.dropna(subset=['self_reported_valence', 'self_reported_arousal', info['v_col'], info['a_col']])
            n = len(valid)
            if n > 0:
                tv = valid['self_reported_valence']
                ta = valid['self_reported_arousal']
                pv = valid[info['v_col']]
                pa = valid[info['a_col']]
                
                # MAE
                mae_v = np.mean(np.abs(tv - pv))
                mae_a = np.mean(np.abs(ta - pa))
                
                # Pearson r
                try: 
                    r_v, _ = pearsonr(tv, pv)
                except: 
                    r_v = np.nan
                try: 
                    r_a, _ = pearsonr(ta, pa)
                except: 
                    r_a = np.nan
                    
                # Accuracy ±1
                acc_v = np.mean(np.abs(tv - pv) <= 1) * 100
                acc_a = np.mean(np.abs(ta - pa) <= 1) * 100
                
                final_records.append({
                    'Model': info['name'],
                    'Dataset': 'In-House (Fitbit)',
                    'N': n,
                    'Valence_MAE': f"{mae_v:.3f}",
                    'Arousal_MAE': f"{mae_a:.3f}",
                    'Valence_Pearson_r': f"{r_v:.3f}" if not np.isnan(r_v) else "N/A",
                    'Arousal_Pearson_r': f"{r_a:.3f}" if not np.isnan(r_a) else "N/A",
                    'Valence_Acc_±1': f"{acc_v:.1f}%",
                    'Arousal_Acc_±1': f"{acc_a:.1f}%"
                })

# ─── 2. Load K-EmoCon Metrics ──────────────────────────────────
if kemocon_metrics_file.exists():
    df_k = pd.read_csv(kemocon_metrics_file)
    for _, row in df_k.iterrows():
        # Clean up model name
        mname = row['model']
        if mname == 'Llama-3.3-70B': mname = 'Llama-3.3-70B-Versatile'
        if mname == 'Mixtral-8x7B': mname = 'Llama-3.1-8B-Instant (Fallback)'
        
        final_records.append({
            'Model': mname,
            'Dataset': 'K-EmoCon (E4)',
            'N': row['n'],
            'Valence_MAE': f"{row['mae_valence']:.3f}",
            'Arousal_MAE': f"{row['mae_arousal']:.3f}",
            'Valence_Pearson_r': f"{row['pearson_r_valence']:.3f}" if pd.notna(row['pearson_r_valence']) else "N/A",
            'Arousal_Pearson_r': f"{row['pearson_r_arousal']:.3f}" if pd.notna(row['pearson_r_arousal']) else "N/A",
            'Valence_Acc_±1': f"{row['accuracy_v_within1']:.1f}%",
            'Arousal_Acc_±1': f"{row['accuracy_a_within1']:.1f}%"
        })

# ─── 3. Save Output ───────────────────────────────────────────
df_final = pd.DataFrame(final_records)
df_final.to_csv('final_validation_table.csv', index=False)

print("\nFINAL VALIDATION METRICS:")
print(df_final.to_string(index=False))
print(f"\n✅ Saved successfully to final_validation_table.csv")
