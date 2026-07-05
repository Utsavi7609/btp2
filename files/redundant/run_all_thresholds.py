import pandas as pd
import os

# This script automates the threshold testing for 0.25, 0.5, 0.75, and 1.0
def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def run_threshold_analysis(threshold):
    results = []
    for dim in ['valence', 'arousal']:
        # Load the 4 required data sources
        p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
        e_df = pd.read_csv(f'final_expressed_{dim}.csv')
        irep_df = pd.read_csv(f'induced_{dim}_summary.csv')
        ipred_df = pd.read_csv(f'inferred_{dim}_summary.csv')

        # Standardize IDs
        for df in [p_df, e_df, irep_df, ipred_df]:
            df['clip_id'] = clean_id(df['clip_id'])

        # Join everything into one Master Table
        m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
        m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id')
        m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
        m.columns = ['id', 'p', 'e', 'i_self', 'i_body']

        # Logic for Matching
        def is_m(a, b): return abs(a - b) <= threshold

        # Count Categories for SELF-REPORTED
        self_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_self), axis=1)])
        self_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_self) and not is_m(r.p, r.e), axis=1)])
        self_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_self), axis=1)])
        self_none = 192 - (self_match + self_feel + self_drift)

        # Count Categories for BODY (PHYSIOLOGICAL)
        body_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_body), axis=1)])
        body_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_body) and not is_m(r.p, r.e), axis=1)])
        body_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_body), axis=1)])
        body_none = 192 - (body_match + body_feel + body_drift)

        results.append({
            'Dimension': dim.capitalize(),
            'Threshold': threshold,
            'Self_Match': self_match,
            'Body_Match': body_match,
            'Self_Feel': self_feel,
            'Body_Feel': body_feel,
            'Self_Drift': self_drift,
            'Body_Drift': body_drift,
            'Self_None': self_none,
            'Body_None': body_none
        })
    return results

# --- EXECUTION ---
all_final_data = []
thresholds = [0.25, 0.5, 0.75, 1.0]

for t in thresholds:
    print(f"Processing Threshold: {t}...")
    all_final_data.extend(run_threshold_analysis(t))

# Create the final Master Table
df_final = pd.DataFrame(all_final_data)

# Print a clean version of the table to the console
for t in thresholds:
    print(f"\n--- TABLE FOR THRESHOLD {t} ---")
    sub = df_final[df_final['Threshold'] == t]
    # Formatting to match the image you provided
    print(f"{'Category':<30} | {'Self (V)':<8} | {'Body (V)':<8} | {'Self (A)':<8} | {'Body (A)':<8}")
    print("-" * 75)
    print(f"{'All Match (P=E=I)':<30} | {sub.iloc[0]['Self_Match']:<8} | {sub.iloc[0]['Body_Match']:<8} | {sub.iloc[1]['Self_Match']:<8} | {sub.iloc[1]['Body_Match']:<8}")
    print(f"{'Feel Perception (P=I, P!=E)':<30} | {sub.iloc[0]['Self_Feel']:<8} | {sub.iloc[0]['Body_Feel']:<8} | {sub.iloc[1]['Self_Feel']:<8} | {sub.iloc[1]['Body_Feel']:<8}")
    print(f"{'Understand/Drift (P=E, P!=I)':<30} | {sub.iloc[0]['Self_Drift']:<8} | {sub.iloc[0]['Body_Drift']:<8} | {sub.iloc[1]['Self_Drift']:<8} | {sub.iloc[1]['Body_Drift']:<8}")
    print(f"{'No Pair Match':<30} | {sub.iloc[0]['Self_None']:<8} | {sub.iloc[0]['Body_None']:<8} | {sub.iloc[1]['Self_None']:<8} | {sub.iloc[1]['Body_None']:<8}")

df_final.to_csv('MASTER_THRESHOLD_RESULTS.csv', index=False)
print("\nSUCCESS: 'MASTER_THRESHOLD_RESULTS.csv' created in your folder.")