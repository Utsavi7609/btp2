import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact

# 1. Load E values
df_e = pd.read_csv("d:/BTP/btp2/salmamaterials/final_expressed_valence_VIDEO.csv")
e_dict = {}
for _, row in df_e.iterrows():
    e_dict[row['clip_id']] = (row['expressed_valence'], row['expressed_arousal'])

df_e_audio = pd.read_csv("d:/BTP/btp2/salmamaterials/final_expressed_valence_AUDIO.csv")
for _, row in df_e_audio.iterrows():
    if row['clip_id'] not in e_dict:
        e_dict[row['clip_id']] = (row['expressed_valence'], row['expressed_arousal'])

df_val = pd.read_csv("D:/BTP/btp2main/clip_responses-export-2026-04-10_18-21-13.csv", sep=";")

df_val['created_at'] = pd.to_datetime(df_val['created_at'])
df_val = df_val.sort_values(by=['created_at'])

# Create a true session ID by checking resets of clip_position 
# or large gaps in time, but a reset to position 1 is a clear new session.
session_ids = []
current_session = 0
prev_pos = None

for _, row in df_val.iterrows():
    pos = row['clip_position']
    if prev_pos is None or pos <= prev_pos:
        # User restarted or new session
        current_session += 1
    session_ids.append(current_session)
    prev_pos = pos

df_val['true_session_id'] = session_ids

# Data structures for statistical testing
maint_deltas = []
drift_deltas = []

hyp9_trans_mirrors = 0
hyp9_trans_totals = 0
hyp9_maint_mirrors = 0
hyp9_maint_totals = 0

for (session, block), group in df_val.groupby(['true_session_id', 'block_type']):
    if block == 'Filler':
        continue
    
    if len(group) < 2:
        continue
        
    clip1 = group[group['expected_role'] == 'Clip1_Stimulus']
    clip2 = group[group['expected_role'] == 'Clip2_Target']
    if clip1.empty or clip2.empty:
        continue
        
    c1 = clip1.iloc[0]
    c2 = clip2.iloc[0]
    
    title2 = str(c2['clip_title']).replace('.mp4', '')
    e2_v, e2_a = e_dict.get(title2, (3.0, 3.0))
    
    # 1. Delta I1 Val
    delta_i1_val = c2['i1_val'] - c1['i1_val']
    abs_delta_i1_val = abs(delta_i1_val)
    
    if block == 'Maintenance_Block':
        maint_deltas.append(abs_delta_i1_val)
    elif block == 'Drift_Block':
        drift_deltas.append(abs_delta_i1_val)
        
    # 2. Mirror Rate
    if block.startswith('Hyp9'):
        i1_c1 = np.array([c1['i1_val'], c1['i1_aro']])
        p_c2 = np.array([c2['p_val'], c2['p_aro']])
        e_c2 = np.array([e2_v, e2_a])
        
        if pd.isna(i1_c1).any() or pd.isna(p_c2).any() or pd.isna(e_c2).any():
            continue
            
        dist_to_i1 = np.linalg.norm(p_c2 - i1_c1)
        dist_to_e = np.linalg.norm(p_c2 - e_c2)
        
        is_mirror = 1 if dist_to_i1 < dist_to_e else 0
        
        if block == 'Hyp9_Transition_Block':
            hyp9_trans_mirrors += is_mirror
            hyp9_trans_totals += 1
        elif block == 'Hyp9_Maintenance_Block':
            hyp9_maint_mirrors += is_mirror
            hyp9_maint_totals += 1

print(f"Total True Sessions (Participants): {df_val['true_session_id'].nunique()}")
print("\n===== Q4 MAINTENANCE VS DRIFT (HYP-1) =====")
print(f"Maintenance Blocks (n={len(maint_deltas)}): Mean abs(Delta I1_val) = {np.mean(maint_deltas):.3f}")
print(f"Drift Blocks       (n={len(drift_deltas)}): Mean abs(Delta I1_val) = {np.mean(drift_deltas):.3f}")

if len(maint_deltas) > 0 and len(drift_deltas) > 0:
    stat, p_val = mannwhitneyu(drift_deltas, maint_deltas, alternative='greater')
    print(f"Mann-Whitney U Test (Drift > Maint): p = {p_val:.4f}")

print("\n===== NEUTRAL MIRRORING (HYP-9) =====")
print(f"Hyp9 Transition Mirror Rate:  {hyp9_trans_mirrors}/{hyp9_trans_totals} ({hyp9_trans_mirrors/max(1, hyp9_trans_totals):.2%})")
print(f"Hyp9 Maintenance Mirror Rate: {hyp9_maint_mirrors}/{hyp9_maint_totals} ({hyp9_maint_mirrors/max(1, hyp9_maint_totals):.2%})")

table = [
    [hyp9_trans_mirrors, hyp9_trans_totals - hyp9_trans_mirrors],
    [hyp9_maint_mirrors, hyp9_maint_totals - hyp9_maint_mirrors]
]
stat, p_val = fisher_exact(table, alternative='greater')
print(f"Fisher's Exact Test (Trans > Maint): p = {p_val:.4f}")

print("\n===== [PENDING FITBIT DATA] (HYP-2 / HYP-6) =====")
print("Stub: Physiological body-mind divergence metrics will be computed here once I2_val and I2_aro are populated.")
