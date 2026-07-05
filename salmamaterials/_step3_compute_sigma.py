"""
Step 3: Compute sigma for all 10 hypotheses across 3 modalities.
Handles empty subsets gracefully for modalities with zero switching.
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = r"d:\BTP\btp2\salmamaterials"
MODALITIES = ['VIDEO', 'AUDIO', 'INTERSECT']
TAU_STRS = ["025", "050", "075", "100"]

def assign_quadrant(val, aro):
    if pd.isna(val) or pd.isna(aro): return "Q0"
    if val == 3.0 or aro == 3.0: return "Q0"
    if val > 3.0 and aro > 3.0: return "Q1"
    if val < 3.0 and aro > 3.0: return "Q2"
    if val < 3.0 and aro < 3.0: return "Q3"
    if val > 3.0 and aro < 3.0: return "Q4"
    return "Q0"

OPPOSING = {('Q1','Q3'), ('Q3','Q1'), ('Q2','Q4'), ('Q4','Q2')}
ADJACENT = {('Q1','Q2'), ('Q2','Q1'), ('Q1','Q4'), ('Q4','Q1'),
            ('Q2','Q3'), ('Q3','Q2'), ('Q3','Q4'), ('Q4','Q3')}

def get_dist(v1, a1, v2, a2):
    return np.sqrt((v1-v2)**2 + (a1-a2)**2)

def safe_frac(subset, col, value):
    """Safely compute fraction of rows where col==value in subset."""
    if len(subset) == 0 or col not in subset.columns:
        return np.nan
    return (subset[col] == value).mean()

def safe_mean(subset, col):
    """Safely compute mean of a column."""
    if len(subset) == 0 or col not in subset.columns:
        return np.nan
    return subset[col].mean()

def filter_quad_pairs(df, pair_set):
    """Filter to rows where (clip1_E_quad, clip2_E_quad) is in pair_set."""
    if len(df) == 0:
        return df.iloc[0:0]  # return empty df with same columns
    mask = df.apply(lambda r: (r['clip1_E_quad'], r['clip2_E_quad']) in pair_set, axis=1)
    return df[mask]

def compute_sigma(modality):
    fpath = os.path.join(BASE_DIR, f"_consecutive_pair_dataset_{modality}.csv")
    print(f"\n{'='*60}")
    print(f"Computing sigma for {modality}")
    print(f"{'='*60}")
    df = pd.read_csv(fpath)
    
    sw_true = df[df['switching'] == True]
    sw_false = df[df['switching'] == False]
    
    results = []
    
    for tau_str in TAU_STRS:
        b_i1 = f'clip2_B_I1_tau{tau_str}'
        b_i2 = f'clip2_B_I2_tau{tau_str}'
        comb = f'clip2_combined_tau{tau_str}'
        c1_b_i1 = f'clip1_B_I1_tau{tau_str}'
        c1_b_i2 = f'clip1_B_I2_tau{tau_str}'
        
        def row(hyp_id, s_sw, n_sw, s_ns, n_ns, s_all=np.nan, n_all=0):
            results.append({
                'hyp_id': hyp_id, 'tau': tau_str, 'modality': modality,
                'sigma_switching': s_sw, 'n_switching': n_sw,
                'sigma_nonswitching': s_ns, 'n_nonswitching': n_ns,
                'sigma_all': s_all, 'n_all': n_all
            })
        
        # Hyp-1: B1 Maintenance
        row('Hyp-1',
            safe_frac(sw_true, b_i1, 'B1'), len(sw_true),
            safe_frac(sw_false, b_i1, 'B1'), len(sw_false),
            safe_frac(df[df['switching'].notna()], b_i1, 'B1'),
            len(df[df['switching'].notna()]))
        
        # Hyp-2: B3+B6 Conscious Denial
        opposing_sw = filter_quad_pairs(sw_true, OPPOSING)
        adjacent_sw = filter_quad_pairs(sw_true, ADJACENT)
        
        row('Hyp-2_opposing',
            safe_frac(opposing_sw, comb, 'B3+B6'), len(opposing_sw),
            safe_frac(sw_false, comb, 'B3+B6'), len(sw_false))
        row('Hyp-2_adjacent',
            safe_frac(adjacent_sw, comb, 'B3+B6'), len(adjacent_sw),
            safe_frac(sw_false, comb, 'B3+B6'), len(sw_false))
        
        # Hyp-3: B2 Projection (neutral clips)
        neutral_q0 = df[df['clip2_E_quad'] == 'Q0']
        if len(neutral_q0) >= 20:
            n_sw_q0 = neutral_q0[neutral_q0['switching'] == True]
            n_ns_q0 = neutral_q0[neutral_q0['switching'] == False]
            row('Hyp-3',
                safe_frac(n_sw_q0, b_i1, 'B2'), len(n_sw_q0),
                safe_frac(n_ns_q0, b_i1, 'B2'), len(n_ns_q0))
        
        # Hyp-3 relaxed
        neutral_relaxed = df[
            (df['clip2_E_val'] - 3.0).abs().le(0.5) &
            (df['clip2_E_aro'] - 3.0).abs().le(0.5)]
        n_sw_r = neutral_relaxed[neutral_relaxed['switching'] == True]
        n_ns_r = neutral_relaxed[neutral_relaxed['switching'] == False]
        row('Hyp-3_relaxed',
            safe_frac(n_sw_r, b_i1, 'B2'), len(n_sw_r),
            safe_frac(n_ns_r, b_i1, 'B2'), len(n_ns_r))
        
        # Hyp-4: Cognitive Noise (AND logic)
        chaos = df[(df[c1_b_i1] == 'B5') & (df[c1_b_i2] == 'B10')]
        baseline = df[(df[c1_b_i1] == 'B1') & (df[c1_b_i2] == 'B6')]
        
        for group_name, group in [('Hyp-4_chaos', chaos), ('Hyp-4_baseline', baseline)]:
            g_sw = group[group['switching'] == True]
            g_ns = group[group['switching'] == False]
            valid_g = group[group['switching'].notna()]
            row(group_name,
                safe_frac(g_sw, b_i1, 'B1'), len(g_sw),
                safe_frac(g_ns, b_i1, 'B1'), len(g_ns),
                safe_frac(valid_g, b_i1, 'B1'), len(valid_g))
        
        # Hyp-5_B4
        row('Hyp-5_B4',
            safe_frac(sw_true, b_i1, 'B4'), len(sw_true),
            safe_frac(sw_false, b_i1, 'B4'), len(sw_false))
        
        # Hyp-5_B9
        row('Hyp-5_B9',
            safe_frac(sw_true, b_i2, 'B9'), len(sw_true),
            safe_frac(sw_false, b_i2, 'B9'), len(sw_false))
        
        # Hyp-6: Diagonal vs Adjacent
        row('Hyp-6_B3B6',
            safe_frac(opposing_sw, comb, 'B3+B6'), len(opposing_sw),
            safe_frac(adjacent_sw, comb, 'B3+B6'), len(adjacent_sw))
        row('Hyp-6_B9',
            safe_frac(opposing_sw, b_i2, 'B9'), len(opposing_sw),
            safe_frac(adjacent_sw, b_i2, 'B9'), len(adjacent_sw))
        
        # Hyp-7: B9 Physiological Override
        row('Hyp-7',
            safe_frac(sw_true, b_i2, 'B9'), len(sw_true),
            safe_frac(sw_false, b_i2, 'B9'), len(sw_false))
        
        # Hyp-8: I1-I2 Convergence
        s_sw_d, s_sw_a, n_sw_d = np.nan, np.nan, 0
        s_ns_d, s_ns_a, n_ns_d = np.nan, np.nan, 0
        
        if len(sw_true) > 0:
            dist_sw = get_dist(sw_true['clip2_I1_val'], sw_true['clip2_I1_aro'],
                               sw_true['clip2_I2_val'], sw_true['clip2_I2_aro'])
            aro_sw = (sw_true['clip2_I1_aro'] - sw_true['clip2_I2_aro']).abs()
            s_sw_d, s_sw_a, n_sw_d = dist_sw.mean(), aro_sw.mean(), len(sw_true)
        
        if len(sw_false) > 0:
            dist_ns = get_dist(sw_false['clip2_I1_val'], sw_false['clip2_I1_aro'],
                               sw_false['clip2_I2_val'], sw_false['clip2_I2_aro'])
            aro_ns = (sw_false['clip2_I1_aro'] - sw_false['clip2_I2_aro']).abs()
            s_ns_d, s_ns_a, n_ns_d = dist_ns.mean(), aro_ns.mean(), len(sw_false)
        
        row('Hyp-8_full', s_sw_d, n_sw_d, s_ns_d, n_ns_d)
        row('Hyp-8_aro', s_sw_a, n_sw_d, s_ns_a, n_ns_d)
        
        # Hyp-9: Neutral B2 Clips Are Mirrors
        for hyp_label, neutral_set in [('Hyp-9', neutral_q0), ('Hyp-9_relaxed', neutral_relaxed)]:
            b2_neutral = neutral_set[neutral_set[b_i1] == 'B2'] if b_i1 in neutral_set.columns else pd.DataFrame()
            if len(b2_neutral) < 10:
                continue
            d_p2_i1c1 = get_dist(b2_neutral['clip2_P_val'], b2_neutral['clip2_P_aro'],
                                 b2_neutral['clip1_I1_val'], b2_neutral['clip1_I1_aro'])
            d_p2_e2 = get_dist(b2_neutral['clip2_P_val'], b2_neutral['clip2_P_aro'],
                               b2_neutral['clip2_E_val'], b2_neutral['clip2_E_aro'])
            mirror = (d_p2_i1c1 < d_p2_e2)
            
            b2_sw = b2_neutral[b2_neutral['switching'] == True]
            b2_ns = b2_neutral[b2_neutral['switching'] == False]
            m_sw = mirror.loc[b2_sw.index] if len(b2_sw) > 0 else pd.Series(dtype=float)
            m_ns = mirror.loc[b2_ns.index] if len(b2_ns) > 0 else pd.Series(dtype=float)
            row(hyp_label,
                m_sw.astype(int).mean() if len(m_sw) > 0 else np.nan, len(b2_sw),
                m_ns.astype(int).mean() if len(m_ns) > 0 else np.nan, len(b2_ns),
                mirror.astype(int).mean(), len(b2_neutral))
        
        # Hyp-10: B10 Suppresses Next Arousal
        b10_c1 = df[df[c1_b_i2] == 'B10']
        b6_c1 = df[df[c1_b_i2] == 'B6']
        
        for grp_name, grp in [('Hyp-10_B10', b10_c1), ('Hyp-10_B6', b6_c1)]:
            g_sw = grp[grp['switching'] == True]
            g_ns = grp[grp['switching'] == False]
            row(grp_name,
                safe_mean(g_sw, 'clip2_I2_aro'), len(g_sw),
                safe_mean(g_ns, 'clip2_I2_aro'), len(g_ns),
                safe_mean(grp, 'clip2_I2_aro'), len(grp))
        
        # MAINTENANCE vs DRIFT PROOF
        valid = df[df['switching'].notna()]
        b1_q1 = valid[(valid[b_i1] == 'B1') & (valid['clip2_E_quad'] == 'Q1')]
        drift = b1_q1[b1_q1['clip1_E_quad'] != 'Q1']
        maint = b1_q1[b1_q1['clip1_E_quad'] == 'Q1']
        
        row('Maintenance_vs_Drift',
            safe_mean(drift, 'delta_I1_val'), len(drift),
            safe_mean(maint, 'delta_I1_val'), len(maint))
    
    res_df = pd.DataFrame(results)
    out = os.path.join(BASE_DIR, f"_sigma_results_{modality}.csv")
    res_df.to_csv(out, index=False)
    
    # Quick sanity check
    h1 = res_df[(res_df['hyp_id'] == 'Hyp-1') & (res_df['tau'] == '050')]
    h7 = res_df[(res_df['hyp_id'] == 'Hyp-7') & (res_df['tau'] == '050')]
    print(f"\n  Sanity check at tau=0.50:")
    if len(h1) > 0:
        print(f"    Hyp-1: sw={h1.iloc[0]['sigma_switching']:.4f} ns={h1.iloc[0]['sigma_nonswitching']:.4f}" if not pd.isna(h1.iloc[0]['sigma_switching']) else f"    Hyp-1: sw=NaN ns={h1.iloc[0]['sigma_nonswitching']:.4f}")
    if len(h7) > 0:
        print(f"    Hyp-7: sw={h7.iloc[0]['sigma_switching']:.4f} ns={h7.iloc[0]['sigma_nonswitching']:.4f}" if not pd.isna(h7.iloc[0]['sigma_switching']) else f"    Hyp-7: sw=NaN ns={h7.iloc[0]['sigma_nonswitching']:.4f}")
    print(f"  Saved to: {out}")

if __name__ == "__main__":
    for mod in MODALITIES:
        compute_sigma(mod)
