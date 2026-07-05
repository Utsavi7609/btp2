import pandas as pd
import numpy as np

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

def get_quad(v, a):
    if pd.isna(v) or pd.isna(a): return "NaN"
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"

df_val = df_val.sort_values(by=['user_id', 'clip_position'])

maintenance_success = 0
maintenance_total = 0
drift_success = 0
drift_total = 0
hyp9_success = 0
hyp9_total = 0

for (user, block), group in df_val.groupby(['user_id', 'block_type']):
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
    
    i1_q1 = get_quad(c1['i1_val'], c1['i1_aro'])
    i1_q2 = get_quad(c2['i1_val'], c2['i1_aro'])
    p_q2 = get_quad(c2['p_val'], c2['p_aro'])
    
    title2 = str(c2['clip_title']).replace('.mp4', '')
    e2_v, e2_a = e_dict.get(title2, (3,3))
    e2_quad = get_quad(e2_v, e2_a)

    if block == 'Maintenance_Block':
        maintenance_total += 1
        # Success if reported emotion is maintained
        if i1_q2 == i1_q1:
            maintenance_success += 1
            
    elif block == 'Drift_Block':
        drift_total += 1
        # Success if it actually drifted away from the previous baseline and matched the target expectation
        if i1_q2 != i1_q1 and (i1_q2 == e2_quad or i1_q2 == p_q2):
            drift_success += 1
            
    elif block.startswith('Hyp9'):
        hyp9_total += 1
        # Success if perceived emotion mirrors induced emotion from the previous clip
        if p_q2 == i1_q1:
            hyp9_success += 1

print("===== VALIDATION PHASE RESULTS =====")
print(f"Maintenance Success (H1):     {maintenance_success}/{maintenance_total} -> {maintenance_success/maintenance_total:.2%}")
print(f"Drift Success (H3 proxy):     {drift_success}/{drift_total} -> {drift_success/drift_total:.2%}")
print(f"Neutral Mirroring Proj (H9):  {hyp9_success}/{hyp9_total} -> {hyp9_success/hyp9_total:.2%}")
