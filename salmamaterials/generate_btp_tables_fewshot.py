import pandas as pd
import numpy as np
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# We use the new Few-Shot summaries we just generated
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]
MODALITY = 'VIDEO'  # Default ground truth source

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def run_analysis(threshold):
    results = []
    for dim in ['valence', 'arousal']:
        try:
            # Inputs (User's Baseline Files)
            p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
            e_df = pd.read_csv(f'final_expressed_{dim}_{MODALITY}.csv')
            irep_df = pd.read_csv(f'induced_{dim}_summary.csv')
            
            # Input (Our New Few-Shot Results)
            ipred_df = pd.read_csv(f'inferred_{dim}_summary_FS.csv')

            # Standardize IDs
            for df in [p_df, e_df, irep_df, ipred_df]:
                df['clip_id'] = clean_id(df['clip_id'])

            # Merge into Master Table
            m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', f'expressed_{dim}']], on='clip_id')
            m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
            m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
            
            m.columns = ['id', 'p', 'e', 'i_self', 'i_body']

            def is_m(a, b): return np.abs(a - b) <= threshold

            # SELF Categorization
            self_match = ((is_m(m.p, m.e)) & (is_m(m.p, m.i_self))).sum()
            self_feel  = ((is_m(m.p, m.i_self)) & (~is_m(m.p, m.e))).sum()
            self_drift = ((is_m(m.p, m.e)) & (~is_m(m.p, m.i_self))).sum()
            self_none  = len(m) - (self_match + self_feel + self_drift)

            # BODY Categorization (Few-Shot LLM)
            body_match = ((is_m(m.p, m.e)) & (is_m(m.p, m.i_body))).sum()
            body_feel  = ((is_m(m.p, m.i_body)) & (~is_m(m.p, m.e))).sum()
            body_drift = ((is_m(m.p, m.e)) & (~is_m(m.p, m.i_body))).sum()
            body_none  = len(m) - (body_match + body_feel + body_drift)

            results.append({
                'Dimension': dim.capitalize(),
                'Match': (self_match, body_match),
                'Feel': (self_feel, body_feel),
                'Drift': (self_drift, body_drift),
                'None': (self_none, body_none)
            })
        except Exception as e:
            print(f"Error processing {dim}: {e}")
            
    return results

print("="*75)
print(f"FEW-SHOT BTP HYPOTHESIS TABLES (Modality: {MODALITY})")
print("="*75)

all_stats = {}
for t in THRESHOLDS:
    all_stats[t] = run_analysis(t)

# ─── SAVE TO CSV ──────────────────────────────────────────────────────────────
csv_rows = []
for t in THRESHOLDS:
    v = all_stats[t][0]
    a = all_stats[t][1]
    csv_rows.append({'Threshold': t, 'Dimension': 'Valence', 'Match_Self': v['Match'][0], 'Match_Body': v['Match'][1], 'Feel_Self': v['Feel'][0], 'Feel_Body': v['Feel'][1], 'Drift_Self': v['Drift'][0], 'Drift_Body': v['Drift'][1], 'None_Self': v['None'][0], 'None_Body': v['None'][1]})
    csv_rows.append({'Threshold': t, 'Dimension': 'Arousal', 'Match_Self': a['Match'][0], 'Match_Body': a['Match'][1], 'Feel_Self': a['Feel'][0], 'Feel_Body': a['Feel'][1], 'Drift_Self': a['Drift'][0], 'Drift_Body': a['Drift'][1], 'None_Self': a['None'][0], 'None_Body': a['None'][1]})

pd.DataFrame(csv_rows).to_csv('btp_fewshot_hypothesis_results.csv', index=False)
print(f"\n✅ Summary saved to btp_fewshot_hypothesis_results.csv")

for t in THRESHOLDS:
    print(f"\n--- TABLE FOR THRESHOLD {t} ---")
    print(f"{'Category':<32} | {'Self (V)':<8} | {'Body (V)':<8} | {'Self (A)':<8} | {'Body (A)':<8}")
    print("-" * 75)
    
    v = all_stats[t][0]
    a = all_stats[t][1]
    
    print(f"{'All Match (P=E=I)':<32} | {v['Match'][0]:<8} | {v['Match'][1]:<8} | {a['Match'][0]:<8} | {a['Match'][1]:<8}")
    print(f"{'Feel Perception (P=I, P!=E)':<32} | {v['Feel'][0]:<8} | {v['Feel'][1]:<8} | {a['Feel'][0]:<8} | {a['Feel'][1]:<8}")
    print(f"{'Understand/Drift (P=E, P!=I)':<32} | {v['Drift'][0]:<8} | {v['Drift'][1]:<8} | {a['Drift'][0]:<8} | {a['Drift'][1]:<8}")
    print(f"{'No Pair Match':<32} | {v['None'][0]:<8} | {v['None'][1]:<8} | {a['None'][0]:<8} | {a['None'][1]:<8}")

print("\n(Note: Body results are based on the latest Few-Shot Llama-70B inference run.)")
