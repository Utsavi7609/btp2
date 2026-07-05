import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr
from sklearn.metrics import cohen_kappa_score

def load(f):
    df = pd.read_csv(f)
    df['clip_id'] = df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
    return df

def is_match(a, b): return abs(a - b) <= 1

def run_joint_analysis(dim):
    # Load all files
    p_df = load(f'perceived_{dim}_summary.csv')
    e_df = load(f'final_expressed_{dim}.csv')
    irep_df = load(f'induced_{dim}_summary.csv')
    ipred_df = load(f'inferred_{dim}_summary.csv')

    # Join into master table
    m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
    m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id')
    m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
    m.columns = ['id', 'p', 'e', 'irep', 'ipred']

    print(f"\n{'='*20} {dim.upper()} MASTER REPORT {'='*20}")

    # --- Q1 & Q2: PERCEPTION & INDUCTION ALIGNMENT ---
    # Does understanding the movie (p=e) help the body feel the movie (i=e)?
    m['pe_match'] = m.apply(lambda r: is_match(r.p, r.e), axis=1)
    m['pi_match'] = m.apply(lambda r: is_match(r.p, r.ipred), axis=1)
    
    table = pd.crosstab(m['pe_match'], m['pi_match'])
    chi2, p_val = chi2_contingency(table)[0:2] if table.shape == (2,2) else (0, 1.0)
    
    print(f"\nQ1 & Q2: Impact of Understanding on Body Induction")
    print(f"Results: {m['pe_match'].sum()} clips show agreement between P and E.")
    print(f"Stat Analysis (Chi-Sq): p-value = {p_val:.4f}")

    # --- Q3: PERMUTATION BREAKDOWN (p, e, ipred) ---
    print(f"\nQ3: Permutation Analysis (Physiological)")
    q3_1 = len(m[(m.pe_match) & (m.pi_match)])
    q3_2 = len(m[(m.pi_match) & (~m.pe_match)])
    q3_3 = len(m[(m.pe_match) & (~m.pi_match)])
    
    print(f"Case 1.1 (p=e=ipred): {q3_1} clips")
    print(f"Case 2 (p=ipred but p!=e): {q3_2} clips")
    print(f"Case 3 (p=e but p!=ipred): {q3_3} clips")

    # --- Q4 & Q5: METHOD COMPARISON (Self-Report vs Physiological) ---
    corr, _ = pearsonr(m.irep, m.ipred)
    kappa = cohen_kappa_score(m.irep.round(), m.ipred.round())
    
    print(f"\nQ4 & Q5: Method Validation (Conscious vs. Bio-Signal)")
    print(f"Sync Count (irep=ipred): {len(m[m.apply(lambda r: is_match(r.irep, r.ipred), axis=1)])} clips")
    print(f"Stat Analysis: Correlation = {corr:.3f}, Cohen's Kappa = {kappa:.3f}")

    # --- Q6: THE GAP ANALYSIS ---
    print(f"\nQ6: The Gap Analysis (When Mind and Body disagree)")
    gap = m[~m.apply(lambda r: is_match(r.irep, r.ipred), axis=1)].copy()
    if not gap.empty:
        f_p = len(gap[gap.apply(lambda r: is_match(r.ipred, r.p), axis=1)])
        f_e = len(gap[gap.apply(lambda r: is_match(r.ipred, r.e), axis=1)])
        print(f"Body follows Perception (P): {f_p} cases")
        print(f"Body follows Ground Truth (E): {f_e} cases")
        print(f"Unique Bio-Response: {len(gap) - f_p - f_e} cases")

run_joint_analysis('valence')
run_joint_analysis('arousal')