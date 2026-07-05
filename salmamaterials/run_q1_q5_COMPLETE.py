"""
Complete Q1-Q5 Analysis for BOTH Video and Audio Sources
Generates comparison table as requested by mentor
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

print("="*70)
print("Q1-Q5 COMPLETE ANALYSIS - VIDEO vs AUDIO COMPARISON")
print("="*70)

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def categorize_emotions(df, tau=1.0):
    """
    Categorize each clip into P=E=I, P=E≠I, P≠E=I, P≠E≠I
    tau = threshold for "agreement" (default 1.0 point)
    """
    results = []
    
    for _, row in df.iterrows():
        P = row['phys_emotion']
        E = row['expr_emotion']
        I = row['self_emotion']
        
        # Check agreements within threshold
        PE_agree = abs(P - E) <= tau
        PI_agree = abs(P - I) <= tau
        EI_agree = abs(E - I) <= tau
        
        if PE_agree and PI_agree and EI_agree:
            category = "P=E=I"
        elif PE_agree and not EI_agree:
            category = "P=E≠I"
        elif not PE_agree and EI_agree:
            category = "P≠E=I"
        else:
            category = "P≠E≠I"
            
        results.append({
            'clip_id': row['clip_id'],
            'P': P,
            'E': E,
            'I': I,
            'category': category,
            'PE_agree': PE_agree,
            'PI_agree': PI_agree,
            'EI_agree': EI_agree
        })
    
    return pd.DataFrame(results)

def run_q1_analysis(df_cat):
    """Q1: Agreement between P and E"""
    total = len(df_cat)
    pe_agree = df_cat['PE_agree'].sum()
    agreement_rate = pe_agree / total * 100
    
    # Cohen's kappa
    p_vals = df_cat['P'].values
    e_vals = df_cat['E'].values
    
    # Discretize to categories for kappa
    p_discrete = np.round(p_vals).astype(int)
    e_discrete = np.round(e_vals).astype(int)
    
    try:
        kappa = cohen_kappa_score(p_discrete, e_discrete)
    except:
        kappa = np.nan
    
    return {
        'agreement_rate': agreement_rate,
        'kappa': kappa,
        'n_agree': pe_agree,
        'n_total': total
    }

def run_q2_analysis(df_cat):
    """Q2: Triple agreement (P=E=I)"""
    total = len(df_cat)
    triple_agree = (df_cat['category'] == 'P=E=I').sum()
    rate = triple_agree / total * 100
    
    return {
        'triple_agreement_rate': rate,
        'n_agree': triple_agree,
        'n_total': total
    }

def run_q3_analysis(df_cat):
    """Q3: Difference between P=E=I vs P≠E≠I groups"""
    pei_group = df_cat[df_cat['category'] == 'P=E=I']
    other_group = df_cat[df_cat['category'] != 'P=E=I']
    
    if len(pei_group) == 0 or len(other_group) == 0:
        return {'significant': False, 'p_value': np.nan}
    
    # Simple t-test on physiological values
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(pei_group['P'], other_group['P'])
    
    return {
        'significant': p_val < 0.05,
        'p_value': p_val,
        't_statistic': t_stat
    }

def run_q4_analysis(df_cat):
    """Q4: Accuracy when P≠E"""
    pe_disagree = df_cat[~df_cat['PE_agree']]
    
    if len(pe_disagree) == 0:
        return {'accuracy': np.nan}
    
    # Check if P or E is closer to I
    pe_disagree = pe_disagree.copy()
    pe_disagree['p_error'] = abs(pe_disagree['P'] - pe_disagree['I'])
    pe_disagree['e_error'] = abs(pe_disagree['E'] - pe_disagree['I'])
    
    p_better = (pe_disagree['p_error'] < pe_disagree['e_error']).sum()
    accuracy = p_better / len(pe_disagree) * 100
    
    return {
        'accuracy': accuracy,
        'p_better': p_better,
        'n_total': len(pe_disagree)
    }

def run_q5_analysis(df_cat):
    """Q5: Independence even when P=E=I"""
    pei_group = df_cat[df_cat['category'] == 'P=E=I']
    
    if len(pei_group) == 0:
        return {'independence_maintained': False}
    
    # Check if there's still variance in P values
    p_variance = pei_group['P'].var()
    independence = p_variance > 0.1  # Arbitrary threshold
    
    return {
        'independence_maintained': independence,
        'p_variance': p_variance,
        'n_samples': len(pei_group)
    }

# ========================================================================
# LOAD DATA
# ========================================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load physiological (P)
df_phys_val = pd.read_csv("inferred_valence_summary.csv")
df_phys_aro = pd.read_csv("inferred_arousal_summary.csv")

# Load self-reported (I)
df_self_val = pd.read_csv("perceived_valence_summary.csv")
df_self_aro = pd.read_csv("perceived_arousal_summary.csv")

# Clean clip IDs (remove .mp4 from self-reported)
df_self_val['clip_id'] = df_self_val['clip_id'].str.replace('.mp4', '', regex=False)
df_self_aro['clip_id'] = df_self_aro['clip_id'].str.replace('.mp4', '', regex=False)

print(f"✅ Loaded physiological emotions: {len(df_phys_val)} valence, {len(df_phys_aro)} arousal")
print(f"✅ Loaded self-reported emotions: {len(df_self_val)} valence, {len(df_self_aro)} arousal")

# ========================================================================
# ANALYSIS FUNCTION
# ========================================================================

def run_full_analysis(source_name, df_expr_val, df_expr_aro):
    """Run complete Q1-Q5 analysis for one source"""
    
    print("\n" + "="*70)
    print(f"ANALYSIS: {source_name}")
    print("="*70)
    
    results = {'source': source_name}
    
    # Merge VALENCE data
    df_val = df_phys_val[['clip_id', 'avg_inferred_valence']].merge(
        df_expr_val[['clip_id', 'expressed_valence']],
        on='clip_id'
    ).merge(
        df_self_val[['clip_id', 'avg_valence']],
        on='clip_id'
    )
    
    df_val.columns = ['clip_id', 'phys_emotion', 'expr_emotion', 'self_emotion']
    
    # Merge AROUSAL data
    df_aro = df_phys_aro[['clip_id', 'avg_inferred_arousal']].merge(
        df_expr_aro[['clip_id', 'expressed_arousal']],
        on='clip_id'
    ).merge(
        df_self_aro[['clip_id', 'avg_arousal']],
        on='clip_id'
    )
    
    df_aro.columns = ['clip_id', 'phys_emotion', 'expr_emotion', 'self_emotion']
    
    print(f"\n✅ Merged data: {len(df_val)} valence clips, {len(df_aro)} arousal clips")
    
    # Categorize
    df_val_cat = categorize_emotions(df_val, tau=1.0)
    df_aro_cat = categorize_emotions(df_aro, tau=1.0)
    
    # Run Q1-Q5 for VALENCE
    print("\n" + "-"*70)
    print("VALENCE ANALYSIS")
    print("-"*70)
    
    q1_val = run_q1_analysis(df_val_cat)
    q2_val = run_q2_analysis(df_val_cat)
    q3_val = run_q3_analysis(df_val_cat)
    q4_val = run_q4_analysis(df_val_cat)
    q5_val = run_q5_analysis(df_val_cat)
    
    print(f"\nQ1: P-E Agreement = {q1_val['agreement_rate']:.1f}% (κ={q1_val['kappa']:.3f})")
    print(f"Q2: P=E=I Rate = {q2_val['triple_agreement_rate']:.1f}%")
    print(f"Q3: Group difference p={q3_val['p_value']:.4f}")
    print(f"Q4: P accuracy when P≠E = {q4_val['accuracy']:.1f}%")
    print(f"Q5: Independence maintained = {q5_val['independence_maintained']}")
    
    # Run Q1-Q5 for AROUSAL
    print("\n" + "-"*70)
    print("AROUSAL ANALYSIS")
    print("-"*70)
    
    q1_aro = run_q1_analysis(df_aro_cat)
    q2_aro = run_q2_analysis(df_aro_cat)
    q3_aro = run_q3_analysis(df_aro_cat)
    q4_aro = run_q4_analysis(df_aro_cat)
    q5_aro = run_q5_analysis(df_aro_cat)
    
    print(f"\nQ1: P-E Agreement = {q1_aro['agreement_rate']:.1f}% (κ={q1_aro['kappa']:.3f})")
    print(f"Q2: P=E=I Rate = {q2_aro['triple_agreement_rate']:.1f}%")
    print(f"Q3: Group difference p={q3_aro['p_value']:.4f}")
    print(f"Q4: P accuracy when P≠E = {q4_aro['accuracy']:.1f}%")
    print(f"Q5: Independence maintained = {q5_aro['independence_maintained']}")
    
    # Store results
    results.update({
        'valence_q1_agreement': q1_val['agreement_rate'],
        'valence_q1_kappa': q1_val['kappa'],
        'valence_q2_triple': q2_val['triple_agreement_rate'],
        'valence_q3_significant': q3_val['significant'],
        'valence_q4_accuracy': q4_val['accuracy'],
        'valence_q5_independent': q5_val['independence_maintained'],
        'arousal_q1_agreement': q1_aro['agreement_rate'],
        'arousal_q1_kappa': q1_aro['kappa'],
        'arousal_q2_triple': q2_aro['triple_agreement_rate'],
        'arousal_q3_significant': q3_aro['significant'],
        'arousal_q4_accuracy': q4_aro['accuracy'],
        'arousal_q5_independent': q5_aro['independence_maintained']
    })
    
    return results

# ========================================================================
# RUN FOR BOTH SOURCES
# ========================================================================

# VIDEO source
df_video_val = pd.read_csv("final_expressed_valence_VIDEO.csv")
df_video_aro = pd.read_csv("final_expressed_arousal_VIDEO.csv")

results_video = run_full_analysis("VIDEO-expressed", df_video_val, df_video_aro)

# AUDIO source
df_audio_val = pd.read_csv("final_expressed_valence_AUDIO.csv")
df_audio_aro = pd.read_csv("final_expressed_arousal_AUDIO.csv")

results_audio = run_full_analysis("AUDIO-predicted", df_audio_val, df_audio_aro)

# ========================================================================
# COMPARISON TABLE
# ========================================================================

print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)

comparison = pd.DataFrame([results_video, results_audio])

print("\n" + comparison.to_string(index=False))

# Save results
comparison.to_csv("Q1_Q5_VIDEO_vs_AUDIO_results.csv", index=False)

print("\n✅ Saved: Q1_Q5_VIDEO_vs_AUDIO_results.csv")

# Create simple summary for mentor
print("\n" + "="*70)
print("SUMMARY FOR MENTOR")
print("="*70)

summary = f"""
METHOD COMPARISON (τ=1.0):

{'Metric':<40} {'VIDEO-expressed':<20} {'AUDIO-predicted':<20}
{'-'*80}
Valence P-E Agreement                   {results_video['valence_q1_agreement']:>6.1f}%            {results_audio['valence_q1_agreement']:>6.1f}%
Arousal P-E Agreement                   {results_video['arousal_q1_agreement']:>6.1f}%            {results_audio['arousal_q1_agreement']:>6.1f}%
Valence P=E=I Rate                      {results_video['valence_q2_triple']:>6.1f}%            {results_audio['valence_q2_triple']:>6.1f}%
Arousal P=E=I Rate                      {results_video['arousal_q2_triple']:>6.1f}%            {results_audio['arousal_q2_triple']:>6.1f}%
"""

print(summary)

with open("COMPARISON_SUMMARY.txt", "w") as f:
    f.write(summary)

print("\n✅ Saved: COMPARISON_SUMMARY.txt")
print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)