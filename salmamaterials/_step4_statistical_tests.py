"""
Step 4: Statistical tests (t-tests) for all hypotheses across 3 modalities.
Handles empty subsets gracefully.
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE_DIR = r"d:\BTP\btp2\salmamaterials"
MODALITIES = ['VIDEO', 'AUDIO', 'INTERSECT']
TAU_STRS = ["025", "050", "075", "100"]

OPPOSING = {('Q1','Q3'), ('Q3','Q1'), ('Q2','Q4'), ('Q4','Q2')}
ADJACENT = {('Q1','Q2'), ('Q2','Q1'), ('Q1','Q4'), ('Q4','Q1'),
            ('Q2','Q3'), ('Q3','Q2'), ('Q3','Q4'), ('Q4','Q3')}

def safe_ttest(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan, "n_too_small"
    t, p = stats.ttest_ind(a, b)
    return t, p, ""

def get_dist(v1, a1, v2, a2):
    return np.sqrt((v1-v2)**2 + (a1-a2)**2)

def safe_col(subset, col, value):
    """Get binary series (col==value) from subset, returns empty if col missing."""
    if len(subset) == 0 or col not in subset.columns:
        return pd.Series(dtype=int)
    return (subset[col] == value).astype(int)

def safe_col_vals(subset, col):
    """Get column values from subset, returns empty if col missing."""
    if len(subset) == 0 or col not in subset.columns:
        return pd.Series(dtype=float)
    return subset[col]

def filter_quad_pairs(df, pair_set):
    if len(df) == 0:
        return df.iloc[0:0]
    mask = df.apply(lambda r: (r['clip1_E_quad'], r['clip2_E_quad']) in pair_set, axis=1)
    return df[mask]

def run_tests(modality):
    fpath = os.path.join(BASE_DIR, f"_consecutive_pair_dataset_{modality}.csv")
    print(f"\n{'='*60}")
    print(f"Statistical tests for {modality}")
    print(f"{'='*60}")
    df = pd.read_csv(fpath)
    sw_t = df[df['switching'] == True]
    sw_f = df[df['switching'] == False]
    
    results = []
    
    for tau_str in TAU_STRS:
        b_i1 = f'clip2_B_I1_tau{tau_str}'
        b_i2 = f'clip2_B_I2_tau{tau_str}'
        comb = f'clip2_combined_tau{tau_str}'
        c1_bi1 = f'clip1_B_I1_tau{tau_str}'
        c1_bi2 = f'clip1_B_I2_tau{tau_str}'
        
        def add(hyp, a_vals, b_vals, s_sw, n_sw, s_ns, n_ns, expected_dir='sw<ns', notes_extra=''):
            a_arr = np.array(a_vals, dtype=float) if len(a_vals) > 0 else np.array([])
            b_arr = np.array(b_vals, dtype=float) if len(b_vals) > 0 else np.array([])
            t, p, note = safe_ttest(a_arr, b_arr)
            if notes_extra:
                note = f"{note}; {notes_extra}" if note else notes_extra
            
            dir_match = np.nan
            if not np.isnan(t):
                if expected_dir == 'sw<ns': dir_match = t < 0
                elif expected_dir == 'sw>ns': dir_match = t > 0
                elif expected_dir == 'none': dir_match = True
                elif expected_dir == 'a<b': dir_match = t < 0
                elif expected_dir == 'a>b': dir_match = t > 0
            
            results.append({
                'hyp_id': hyp, 'tau': tau_str, 'modality': modality,
                't_stat': t, 'p_value': p,
                'significant': p < 0.05 if not np.isnan(p) else False,
                'sigma_switching': s_sw, 'n_switching': n_sw,
                'sigma_nonswitching': s_ns, 'n_nonswitching': n_ns,
                'effect_direction_matches': dir_match, 'notes': note
            })
        
        # Hyp-1
        a = safe_col(sw_t, b_i1, 'B1')
        b = safe_col(sw_f, b_i1, 'B1')
        add('Hyp-1', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'sw<ns')
        
        # Hyp-2
        opp = filter_quad_pairs(sw_t, OPPOSING)
        a = safe_col(opp, comb, 'B3+B6')
        b = safe_col(sw_f, comb, 'B3+B6')
        add('Hyp-2', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'sw>ns')
        
        # Hyp-3 relaxed
        neut = df[(df['clip2_E_val'] - 3.0).abs().le(0.5) & (df['clip2_E_aro'] - 3.0).abs().le(0.5)]
        a = safe_col(neut[neut['switching']==True], b_i1, 'B2')
        b = safe_col(neut[neut['switching']==False], b_i1, 'B2')
        add('Hyp-3_relaxed', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'sw<ns')
        
        # Hyp-4
        chaos = df[(df[c1_bi1] == 'B5') & (df[c1_bi2] == 'B10')]
        baseline = df[(df[c1_bi1] == 'B1') & (df[c1_bi2] == 'B6')]
        a = safe_col(chaos, b_i1, 'B1')
        b = safe_col(baseline, b_i1, 'B1')
        add('Hyp-4_chaos_vs_baseline', a, b,
            a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'a<b')
        
        # Hyp-5_B4
        a = safe_col(sw_t, b_i1, 'B4')
        b = safe_col(sw_f, b_i1, 'B4')
        add('Hyp-5_B4', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'none')
        
        # Hyp-5_B9
        a = safe_col(sw_t, b_i2, 'B9')
        b = safe_col(sw_f, b_i2, 'B9')
        add('Hyp-5_B9', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'none', 'same_as_Hyp-7_if_sig')
        
        # Hyp-6
        adj = filter_quad_pairs(sw_t, ADJACENT)
        a = safe_col(opp, comb, 'B3+B6')
        b = safe_col(adj, comb, 'B3+B6')
        add('Hyp-6', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'a>b')
        
        # Hyp-6_B9
        a = safe_col(opp, b_i2, 'B9')
        b = safe_col(adj, b_i2, 'B9')
        add('Hyp-6_B9', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'a>b')
        
        # Hyp-7
        a = safe_col(sw_t, b_i2, 'B9')
        b = safe_col(sw_f, b_i2, 'B9')
        add('Hyp-7', a, b, a.mean() if len(a)>0 else np.nan, len(a),
            b.mean() if len(b)>0 else np.nan, len(b), 'sw>ns')
        
        # Hyp-8_full
        if len(sw_t) > 0 and len(sw_f) > 0:
            dist_sw = get_dist(sw_t['clip2_I1_val'], sw_t['clip2_I1_aro'],
                               sw_t['clip2_I2_val'], sw_t['clip2_I2_aro'])
            dist_ns = get_dist(sw_f['clip2_I1_val'], sw_f['clip2_I1_aro'],
                               sw_f['clip2_I2_val'], sw_f['clip2_I2_aro'])
            add('Hyp-8_full', dist_sw, dist_ns, dist_sw.mean(), len(dist_sw),
                dist_ns.mean(), len(dist_ns), 'sw>ns')
        else:
            add('Hyp-8_full', [], [], np.nan, 0, np.nan, 0, 'sw>ns', 'no_switching_data')
        
        # Hyp-8_aro
        if len(sw_t) > 0 and len(sw_f) > 0:
            aro_sw = (sw_t['clip2_I1_aro'] - sw_t['clip2_I2_aro']).abs()
            aro_ns = (sw_f['clip2_I1_aro'] - sw_f['clip2_I2_aro']).abs()
            add('Hyp-8_aro', aro_sw, aro_ns, aro_sw.mean(), len(aro_sw),
                aro_ns.mean(), len(aro_ns), 'sw>ns')
        else:
            add('Hyp-8_aro', [], [], np.nan, 0, np.nan, 0, 'sw>ns', 'no_switching_data')
        
        # Hyp-9
        neut_b2 = neut[neut[b_i1] == 'B2'] if b_i1 in neut.columns else pd.DataFrame()
        if len(neut_b2) >= 10:
            d1 = get_dist(neut_b2['clip2_P_val'], neut_b2['clip2_P_aro'],
                         neut_b2['clip1_I1_val'], neut_b2['clip1_I1_aro'])
            d2 = get_dist(neut_b2['clip2_P_val'], neut_b2['clip2_P_aro'],
                         neut_b2['clip2_E_val'], neut_b2['clip2_E_aro'])
            mirror = (d1 < d2).astype(int)
            sw_m = mirror.loc[neut_b2[neut_b2['switching']==True].index]
            ns_m = mirror.loc[neut_b2[neut_b2['switching']==False].index]
            add('Hyp-9', sw_m, ns_m,
                sw_m.mean() if len(sw_m)>0 else np.nan, len(sw_m),
                ns_m.mean() if len(ns_m)>0 else np.nan, len(ns_m), 'sw<ns')
        else:
            add('Hyp-9', [], [], np.nan, 0, np.nan, 0, 'sw<ns', f'n_b2_neutral={len(neut_b2)}')
        
        # Hyp-10
        b10 = safe_col_vals(df[df[c1_bi2]=='B10'], 'clip2_I2_aro')
        b6 = safe_col_vals(df[df[c1_bi2]=='B6'], 'clip2_I2_aro')
        add('Hyp-10', b10, b6,
            b10.mean() if len(b10)>0 else np.nan, len(b10),
            b6.mean() if len(b6)>0 else np.nan, len(b6), 'a<b')
        
        # Maintenance vs Drift
        valid = df[df['switching'].notna()]
        b1_q1 = valid[(valid[b_i1]=='B1') & (valid['clip2_E_quad']=='Q1')]
        drift_v = safe_col_vals(b1_q1[b1_q1['clip1_E_quad']!='Q1'], 'delta_I1_val')
        maint_v = safe_col_vals(b1_q1[b1_q1['clip1_E_quad']=='Q1'], 'delta_I1_val')
        add('Maintenance_vs_Drift', drift_v, maint_v,
            drift_v.mean() if len(drift_v)>0 else np.nan, len(drift_v),
            maint_v.mean() if len(maint_v)>0 else np.nan, len(maint_v), 'none')
    
    res_df = pd.DataFrame(results)
    out = os.path.join(BASE_DIR, f"_statistical_tests_{modality}.csv")
    res_df.to_csv(out, index=False)
    
    sig = res_df[res_df['significant'] == True]
    print(f"  {len(sig)} significant results out of {len(res_df)} tests.")
    print(f"  Saved to: {out}")

if __name__ == "__main__":
    for mod in MODALITIES:
        run_tests(mod)
