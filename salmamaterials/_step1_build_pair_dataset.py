"""
Step 1: Build Consecutive Pair Datasets for THREE modalities (VIDEO, AUDIO, INTERSECT).
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = r"d:\BTP\btp2\salmamaterials"
ORDER_FILE = r"d:\BTP\btp2\_chronological_clip_orders.csv"

MODALITIES = ['VIDEO', 'AUDIO', 'INTERSECT']

normalize = lambda x: str(x).replace('.mp4', '').strip()

def assign_quadrant(val, aro):
    if pd.isna(val) or pd.isna(aro):
        return "Q0"
    if val == 3.0 or aro == 3.0:
        return "Q0"
    if val > 3.0 and aro > 3.0: return "Q1"
    if val < 3.0 and aro > 3.0: return "Q2"
    if val < 3.0 and aro < 3.0: return "Q3"
    if val > 3.0 and aro < 3.0: return "Q4"
    return "Q0"

def load_e_data(modality):
    """Load E values for given modality. Returns dict: clip_id -> (e_val, e_aro)"""
    if modality == 'VIDEO':
        df = pd.read_csv(os.path.join(BASE_DIR, "final_expressed_valence_VIDEO.csv"))
        df['clip_id'] = df['clip_id'].apply(normalize)
        df = df.drop_duplicates('clip_id')
        return df.set_index('clip_id')[['expressed_valence', 'expressed_arousal']].rename(
            columns={'expressed_valence': 'e_v', 'expressed_arousal': 'e_a'}).to_dict('index')
    
    elif modality == 'AUDIO':
        df = pd.read_csv(os.path.join(BASE_DIR, "final_expressed_valence_AUDIO.csv"))
        df['clip_id'] = df['clip_id'].apply(normalize)
        df = df.drop_duplicates('clip_id')
        return df.set_index('clip_id')[['expressed_valence', 'expressed_arousal']].rename(
            columns={'expressed_valence': 'e_v', 'expressed_arousal': 'e_a'}).to_dict('index')
    
    elif modality == 'INTERSECT':
        # Load both VIDEO and AUDIO
        vid = pd.read_csv(os.path.join(BASE_DIR, "final_expressed_valence_VIDEO.csv"))
        aud = pd.read_csv(os.path.join(BASE_DIR, "final_expressed_valence_AUDIO.csv"))
        vid['clip_id'] = vid['clip_id'].apply(normalize)
        aud['clip_id'] = aud['clip_id'].apply(normalize)
        vid = vid.drop_duplicates('clip_id').set_index('clip_id')
        aud = aud.drop_duplicates('clip_id').set_index('clip_id')
        
        result = {}
        for cid in vid.index:
            if cid not in aud.index:
                continue
            v_val, v_aro = vid.loc[cid, 'expressed_valence'], vid.loc[cid, 'expressed_arousal']
            a_val, a_aro = aud.loc[cid, 'expressed_valence'], aud.loc[cid, 'expressed_arousal']
            q_vid = assign_quadrant(v_val, v_aro)
            q_aud = assign_quadrant(a_val, a_aro)
            if q_vid == q_aud and q_vid != 'Q0':
                result[cid] = {'e_v': v_val, 'e_a': v_aro}  # Use VIDEO values
        return result

def load_other_data():
    """Load P, I1, I2 data. Returns dict: clip_id -> {p_v, p_a, i1_v, i1_a, i2_v, i2_a}"""
    pv = pd.read_csv(os.path.join(BASE_DIR, "perceived_valence_summary.csv"))
    pa = pd.read_csv(os.path.join(BASE_DIR, "perceived_arousal_summary.csv"))
    i1v = pd.read_csv(os.path.join(BASE_DIR, "induced_valence_summary.csv"))
    i1a = pd.read_csv(os.path.join(BASE_DIR, "induced_arousal_summary.csv"))
    i2v = pd.read_csv(os.path.join(BASE_DIR, "_inferred_valence_summary_llama.csv"))
    i2a = pd.read_csv(os.path.join(BASE_DIR, "_inferred_arousal_summary_llama.csv"))

    for df in [pv, pa, i1v, i1a, i2v, i2a]:
        df['clip_id'] = df['clip_id'].apply(normalize)
        
    result = {}
    all_ids = set(pv['clip_id']) | set(i1v['clip_id']) | set(i2v['clip_id'])
    
    pv_d = pv.drop_duplicates('clip_id').set_index('clip_id')['avg_valence'].to_dict()
    pa_d = pa.drop_duplicates('clip_id').set_index('clip_id')['avg_arousal'].to_dict()
    i1v_d = i1v.drop_duplicates('clip_id').set_index('clip_id')['avg_valence'].to_dict()
    i1a_d = i1a.drop_duplicates('clip_id').set_index('clip_id')['avg_arousal'].to_dict()
    i2v_d = i2v.drop_duplicates('clip_id').set_index('clip_id')['avg_inferred_valence'].to_dict()
    i2a_d = i2a.drop_duplicates('clip_id').set_index('clip_id')['avg_inferred_arousal'].to_dict()
    
    for cid in all_ids:
        result[cid] = {
            'p_v': pv_d.get(cid, np.nan), 'p_a': pa_d.get(cid, np.nan),
            'i1_v': i1v_d.get(cid, np.nan), 'i1_a': i1a_d.get(cid, np.nan),
            'i2_v': i2v_d.get(cid, np.nan), 'i2_a': i2a_d.get(cid, np.nan),
        }
    return result

def build_pairs(modality):
    print(f"\n{'='*60}")
    print(f"Building pair dataset for modality: {modality}")
    print(f"{'='*60}")
    
    e_data = load_e_data(modality)
    other_data = load_other_data()
    order_df = pd.read_csv(ORDER_FILE)
    users = order_df.columns.tolist()
    
    print(f"  E data clips: {len(e_data)}")
    print(f"  Other data clips: {len(other_data)}")
    
    all_pairs = []
    for user in users:
        sequence = order_df[user].dropna().tolist()
        for i in range(len(sequence) - 1):
            c1_raw, c2_raw = sequence[i], sequence[i+1]
            c1, c2 = normalize(c1_raw), normalize(c2_raw)
            
            # E values: NaN if clip not in this modality's E data
            e1 = e_data.get(c1, {'e_v': np.nan, 'e_a': np.nan})
            e2 = e_data.get(c2, {'e_v': np.nan, 'e_a': np.nan})
            o1 = other_data.get(c1, {'p_v': np.nan, 'p_a': np.nan, 'i1_v': np.nan, 'i1_a': np.nan, 'i2_v': np.nan, 'i2_a': np.nan})
            o2 = other_data.get(c2, {'p_v': np.nan, 'p_a': np.nan, 'i1_v': np.nan, 'i1_a': np.nan, 'i2_v': np.nan, 'i2_a': np.nan})
            
            c1_eq = assign_quadrant(e1['e_v'], e1['e_a'])
            c2_eq = assign_quadrant(e2['e_v'], e2['e_a'])
            
            # Switching: NaN if either is Q0
            if c1_eq == 'Q0' or c2_eq == 'Q0':
                switching = np.nan
            else:
                switching = c1_eq != c2_eq
            
            row = {
                'clip1_title': c1_raw, 'clip2_title': c2_raw, 'user_id': user,
                'clip1_E_val': e1['e_v'], 'clip1_E_aro': e1['e_a'],
                'clip1_P_val': o1['p_v'], 'clip1_P_aro': o1['p_a'],
                'clip1_I1_val': o1['i1_v'], 'clip1_I1_aro': o1['i1_a'],
                'clip1_I2_val': o1['i2_v'], 'clip1_I2_aro': o1['i2_a'],
                'clip2_E_val': e2['e_v'], 'clip2_E_aro': e2['e_a'],
                'clip2_P_val': o2['p_v'], 'clip2_P_aro': o2['p_a'],
                'clip2_I1_val': o2['i1_v'], 'clip2_I1_aro': o2['i1_a'],
                'clip2_I2_val': o2['i2_v'], 'clip2_I2_aro': o2['i2_a'],
                'clip1_E_quad': c1_eq, 'clip2_E_quad': c2_eq,
                'switching': switching,
                'delta_I1_val': o2['i1_v'] - o1['i1_v'],
                'delta_I1_aro': o2['i1_a'] - o1['i1_a'],
                'delta_I2_val': o2['i2_v'] - o1['i2_v'],
                'delta_I2_aro': o2['i2_a'] - o1['i2_a'],
            }
            all_pairs.append(row)
    
    df = pd.DataFrame(all_pairs)
    out = os.path.join(BASE_DIR, f"_consecutive_pair_dataset_{modality}.csv")
    df.to_csv(out, index=False)
    
    # Verification
    print(f"\n  Total rows: {len(df)}")
    print(f"  Clip1 Quadrant Distribution:")
    print(df['clip1_E_quad'].value_counts().to_string())
    q3_count = (df['clip1_E_quad'] == 'Q3').sum()
    if q3_count < 50:
        print(f"  WARNING: Q3 count ({q3_count}) < 50!")
    valid = df['switching'].notna()
    print(f"  Switching=True: {(df.loc[valid, 'switching'] == True).sum()}")
    print(f"  Switching=False: {(df.loc[valid, 'switching'] == False).sum()}")
    print(f"  Switching=NaN (Q0 boundary): {(~valid).sum()}")
    print(f"  Saved to: {out}")

if __name__ == "__main__":
    for mod in MODALITIES:
        build_pairs(mod)
