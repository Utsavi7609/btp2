"""
Step 2: Assign B1-B10 bucket labels to all 3 modality pair datasets.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = r"d:\BTP\btp2\salmamaterials"
MODALITIES = ['VIDEO', 'AUDIO', 'INTERSECT']
THRESHOLDS = [0.25, 0.50, 0.75, 1.00]

def are_equal(v1, a1, v2, a2, tau):
    if pd.isna(v1) or pd.isna(a1) or pd.isna(v2) or pd.isna(a2):
        return False
    return (abs(v1 - v2) <= tau) and (abs(a1 - a2) <= tau)

def get_bucket(e_v, e_a, p_v, p_a, i_v, i_a, tau, mode='I1'):
    pe = are_equal(p_v, p_a, e_v, e_a, tau)
    pi = are_equal(p_v, p_a, i_v, i_a, tau)
    ei = are_equal(e_v, e_a, i_v, i_a, tau)
    
    if mode == 'I1':
        if pe and pi: return "B1"
        if pi and not pe: return "B2"
        if pe and not pi: return "B3"
        if ei and not pe: return "B4"
        return "B5"
    else:
        if pe and pi: return "B6"
        if pi and not pe: return "B7"
        if pe and not pi: return "B8"
        if ei and not pe: return "B9"
        return "B10"

def assign_buckets(modality):
    fpath = os.path.join(BASE_DIR, f"_consecutive_pair_dataset_{modality}.csv")
    print(f"\nAssigning buckets for {modality}...")
    df = pd.read_csv(fpath)
    
    for tau in THRESHOLDS:
        tau_str = str(int(tau * 100)).zfill(3)
        
        for clip_prefix in ['clip1', 'clip2']:
            ev, ea = f'{clip_prefix}_E_val', f'{clip_prefix}_E_aro'
            pv, pa = f'{clip_prefix}_P_val', f'{clip_prefix}_P_aro'
            i1v, i1a = f'{clip_prefix}_I1_val', f'{clip_prefix}_I1_aro'
            i2v, i2a = f'{clip_prefix}_I2_val', f'{clip_prefix}_I2_aro'
            
            df[f'{clip_prefix}_B_I1_tau{tau_str}'] = df.apply(
                lambda r: get_bucket(r[ev], r[ea], r[pv], r[pa], r[i1v], r[i1a], tau, 'I1'), axis=1)
            df[f'{clip_prefix}_B_I2_tau{tau_str}'] = df.apply(
                lambda r: get_bucket(r[ev], r[ea], r[pv], r[pa], r[i2v], r[i2a], tau, 'I2'), axis=1)
        
        df[f'clip2_combined_tau{tau_str}'] = (
            df[f'clip2_B_I1_tau{tau_str}'] + "+" + df[f'clip2_B_I2_tau{tau_str}'])
    
    df.to_csv(fpath, index=False)
    print(f"  Updated {len(df)} rows with bucket labels.")
    
    # Quick verification at tau=050
    tau_str = "050"
    print(f"  Clip2 B_I1 distribution (tau=0.50):")
    print(df[f'clip2_B_I1_tau{tau_str}'].value_counts().to_string())

if __name__ == "__main__":
    for mod in MODALITIES:
        assign_buckets(mod)
