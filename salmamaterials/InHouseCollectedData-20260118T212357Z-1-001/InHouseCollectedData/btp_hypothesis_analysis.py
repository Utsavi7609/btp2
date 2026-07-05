import pandas as pd
import numpy as np
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
CLEAN_DATASET = "clean_dataset.csv"
LLM_RESULTS   = "llm_results_llama_fewshot.csv"
OUTPUT_TABLE  = "btp_hypothesis_results.csv"

# Thresholds to test
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
if not os.path.exists(CLEAN_DATASET) or not os.path.exists(LLM_RESULTS):
    print("Error: Missing data files.")
    exit(1)

df_clean = pd.read_csv(CLEAN_DATASET)
df_llm   = pd.read_csv(LLM_RESULTS)

# Standardize IDs for merging
df_clean['clip_id'] = clean_id(df_clean['clip_title'])
df_llm['clip_id']   = clean_id(df_llm['clip_title'])

# Merge
# We'll use the LLM results as the primary driver for 'Body' data (i_body)
# and the clean_dataset for Ground Truth (e), Perception (p), and Induced (i_self)
m = pd.merge(
    df_clean[['participant', 'clip_id', 'perceived_valence', 'perceived_arousal', 'expressed_valence', 'expressed_arousal', 'self_reported_valence', 'self_reported_arousal']],
    df_llm[['participant', 'clip_id', 'inferred_valence', 'inferred_arousal']],
    on=['participant', 'clip_id']
)

print(f"✅ Merged {len(m)} rows for hypothesis analysis.")

# ─── CATEGORIZATION LOGIC ─────────────────────────────────────────────────────
def get_categories(df, threshold, dim):
    # Mapping to standard names used in run_all_thresholds.py
    p = df[f'perceived_{dim}'].astype(float)
    e = df[f'expressed_{dim}'].astype(float)
    i_self = df[f'self_reported_{dim}'].astype(float)
    i_body = df[f'inferred_{dim}'].astype(float)

    def is_m(a, b): return np.abs(a - b) <= threshold

    # Counts for SELF
    self_match = ((is_m(p, e)) & (is_m(p, i_self))).sum()
    self_feel  = ((is_m(p, i_self)) & (~is_m(p, e))).sum()
    self_drift = ((is_m(p, e)) & (~is_m(p, i_self))).sum()
    self_none  = len(df) - (self_match + self_feel + self_drift)

    # Counts for BODY
    body_match = ((is_m(p, e)) & (is_m(p, i_body))).sum()
    body_feel  = ((is_m(p, i_body)) & (~is_m(p, e))).sum()
    body_drift = ((is_m(p, e)) & (~is_m(p, i_body))).sum()
    body_none  = len(df) - (body_match + body_feel + body_drift)

    return {
        'Self_Match': self_match, 'Body_Match': body_match,
        'Self_Feel': self_feel,   'Body_Feel': body_feel,
        'Self_Drift': self_drift, 'Body_Drift': body_drift,
        'Self_None': self_none,   'Body_None': body_none
    }

# ─── RUN ANALYSIS ─────────────────────────────────────────────────────────────
final_data = []

for t in THRESHOLDS:
    v_stats = get_categories(m, t, 'valence')
    a_stats = get_categories(m, t, 'arousal')
    
    final_data.append({
        'Threshold': t,
        'Valence': v_stats,
        'Arousal': a_stats
    })

# ─── PRINT TABLES (IMAGE STYLE) ───────────────────────────────────────────────
for entry in final_data:
    t = entry['Threshold']
    v = entry['Valence']
    a = entry['Arousal']
    
    print(f"\n--- TABLE FOR THRESHOLD {t} ---")
    print(f"{'Category':<30} | {'Self (V)':<8} | {'Body (V)':<8} | {'Self (A)':<8} | {'Body (A)':<8}")
    print("-" * 75)
    
    print(f"{'All Match (P=E=I)':<30} | {v['Self_Match']:<8} | {v['Body_Match']:<8} | {a['Self_Match']:<8} | {a['Body_Match']:<8}")
    print(f"{'Feel Perception (P=I, P!=E)':<30} | {v['Self_Feel']:<8} | {v['Body_Feel']:<8} | {a['Self_Feel']:<8} | {a['Body_Feel']:<8}")
    print(f"{'Understand/Drift (P=E, P!=I)':<30} | {v['Self_Drift']:<8} | {v['Body_Drift']:<8} | {a['Self_Drift']:<8} | {a['Body_Drift']:<8}")
    print(f"{'No Pair Match':<30} | {v['Self_None']:<8} | {v['Body_None']:<8} | {a['Self_None']:<8} | {a['Body_None']:<8}")

# Save to CSV for the user
df_out = []
for entry in final_data:
    t = entry['Threshold']
    for dim in ['Valence', 'Arousal']:
        row = {'Threshold': t, 'Dimension': dim}
        row.update(entry[dim])
        df_out.append(row)

pd.DataFrame(df_out).to_csv(OUTPUT_TABLE, index=False)
print(f"\n✅ Summary saved to {OUTPUT_TABLE}")
