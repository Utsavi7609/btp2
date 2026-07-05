# """
# Complete Threshold Analysis for Both VIDEO and AUDIO Sources
# Generates 8 tables: 4 thresholds × 2 sources
# """

# import pandas as pd
# import numpy as np
# from scipy.stats import chi2_contingency, ttest_ind
# from sklearn.metrics import cohen_kappa_score

# print("="*70)
# print("THRESHOLD ANALYSIS - VIDEO vs AUDIO")
# print("="*70)

# # ========================================================================
# # LOAD DATA
# # ========================================================================

# print("\nLoading data...")

# # Physiological (P)
# df_phys_val = pd.read_csv("inferred_valence_summary.csv")
# df_phys_aro = pd.read_csv("inferred_arousal_summary.csv")

# # Self-reported (I)
# df_self_val = pd.read_csv("perceived_valence_summary.csv")
# df_self_aro = pd.read_csv("perceived_arousal_summary.csv")

# # Clean clip IDs
# df_self_val['clip_id'] = df_self_val['clip_id'].str.replace('.mp4', '', regex=False)
# df_self_aro['clip_id'] = df_self_aro['clip_id'].str.replace('.mp4', '', regex=False)

# print(f"✅ Loaded physiological and self-reported data")

# # ========================================================================
# # CATEGORIZATION FUNCTION
# # ========================================================================

# def categorize_emotions(df_merged, tau):
#     """
#     Categorize emotions into 4 groups based on threshold
#     """
#     df = df_merged.copy()
    
#     # Calculate agreements
#     df['PE_agree'] = abs(df['P'] - df['E']) <= tau
#     df['PI_agree'] = abs(df['P'] - df['I']) <= tau
#     df['EI_agree'] = abs(df['E'] - df['I']) <= tau
    
#     # Categorize
#     def get_category(row):
#         if row['PE_agree'] and row['PI_agree']:
#             return 'P=E=I'
#         elif row['PE_agree'] and not row['PI_agree']:
#             return 'P=E≠I'
#         elif not row['PE_agree'] and row['EI_agree']:
#             return 'P≠E=I'
#         else:
#             return 'P≠E≠I'
    
#     df['category'] = df.apply(get_category, axis=1)
    
#     return df

# # ========================================================================
# # ANALYSIS FUNCTION
# # ========================================================================

# def run_threshold_analysis(source_name, df_expr_val, df_expr_aro, tau):
#     """
#     Run analysis for one source at one threshold
#     """
    
#     # Merge VALENCE
#     df_val = df_phys_val[['clip_id', 'avg_inferred_valence']].merge(
#         df_expr_val[['clip_id', 'expressed_valence']],
#         on='clip_id'
#     ).merge(
#         df_self_val[['clip_id', 'avg_valence']],
#         on='clip_id'
#     )
#     df_val.columns = ['clip_id', 'P', 'E', 'I']
    
#     # Merge AROUSAL
#     df_aro = df_phys_aro[['clip_id', 'avg_inferred_arousal']].merge(
#         df_expr_aro[['clip_id', 'expressed_arousal']],
#         on='clip_id'
#     ).merge(
#         df_self_aro[['clip_id', 'avg_arousal']],
#         on='clip_id'
#     )
#     df_aro.columns = ['clip_id', 'P', 'E', 'I']
    
#     # Categorize
#     df_val_cat = categorize_emotions(df_val, tau)
#     df_aro_cat = categorize_emotions(df_aro, tau)
    
#     # Count categories
#     val_counts = df_val_cat['category'].value_counts()
#     aro_counts = df_aro_cat['category'].value_counts()
    
#     # Calculate metrics for each category
#     results = []
    
#     for category in ['P=E=I', 'P=E≠I', 'P≠E=I', 'P≠E≠I']:
#         # Valence
#         val_subset = df_val_cat[df_val_cat['category'] == category]
#         val_n = len(val_subset)
#         val_pct = (val_n / len(df_val_cat) * 100) if len(df_val_cat) > 0 else 0
        
#         # Arousal
#         aro_subset = df_aro_cat[df_aro_cat['category'] == category]
#         aro_n = len(aro_subset)
#         aro_pct = (aro_n / len(df_aro_cat) * 100) if len(df_aro_cat) > 0 else 0
        
#         results.append({
#             'Source': source_name,
#             'Threshold': tau,
#             'Category': category,
#             'Valence_N': val_n,
#             'Valence_Pct': val_pct,
#             'Arousal_N': aro_n,
#             'Arousal_Pct': aro_pct
#         })
    
#     # Overall agreement rates
#     val_pe_agree = df_val_cat['PE_agree'].sum() / len(df_val_cat) * 100
#     aro_pe_agree = df_aro_cat['PE_agree'].sum() / len(df_aro_cat) * 100
    
#     # Cohen's kappa
#     try:
#         val_kappa = cohen_kappa_score(
#             np.round(df_val_cat['P']).astype(int),
#             np.round(df_val_cat['E']).astype(int)
#         )
#     except:
#         val_kappa = np.nan
    
#     try:
#         aro_kappa = cohen_kappa_score(
#             np.round(df_aro_cat['P']).astype(int),
#             np.round(df_aro_cat['E']).astype(int)
#         )
#     except:
#         aro_kappa = np.nan
    
#     summary = {
#         'Source': source_name,
#         'Threshold': tau,
#         'Valence_PE_Agreement': val_pe_agree,
#         'Valence_Kappa': val_kappa,
#         'Arousal_PE_Agreement': aro_pe_agree,
#         'Arousal_Kappa': aro_kappa,
#         'Total_Clips': len(df_val_cat)
#     }
    
#     return pd.DataFrame(results), summary

# # ========================================================================
# # RUN FOR ALL THRESHOLDS AND BOTH SOURCES
# # ========================================================================

# thresholds = [0.25, 0.5, 0.75, 1.0]

# # Load VIDEO and AUDIO expressed emotions
# df_video_val = pd.read_csv("final_expressed_valence_VIDEO.csv")
# df_video_aro = pd.read_csv("final_expressed_arousal_VIDEO.csv")

# df_audio_val = pd.read_csv("final_expressed_valence_AUDIO.csv")
# df_audio_aro = pd.read_csv("final_expressed_arousal_AUDIO.csv")

# all_category_results = []
# all_summaries = []

# print("\n" + "="*70)
# print("RUNNING THRESHOLD ANALYSIS")
# print("="*70)

# for tau in thresholds:
#     print(f"\n{'='*70}")
#     print(f"THRESHOLD τ = {tau}")
#     print(f"{'='*70}")
    
#     # VIDEO source
#     print(f"\n  VIDEO-expressed (τ={tau})...")
#     video_results, video_summary = run_threshold_analysis(
#         "VIDEO", df_video_val, df_video_aro, tau
#     )
#     all_category_results.append(video_results)
#     all_summaries.append(video_summary)
    
#     # AUDIO source
#     print(f"  AUDIO-predicted (τ={tau})...")
#     audio_results, audio_summary = run_threshold_analysis(
#         "AUDIO", df_audio_val, df_audio_aro, tau
#     )
#     all_category_results.append(audio_results)
#     all_summaries.append(audio_summary)
    
#     # Show results for this threshold
#     print(f"\n  VIDEO Results (τ={tau}):")
#     print(video_results.to_string(index=False))
    
#     print(f"\n  AUDIO Results (τ={tau}):")
#     print(audio_results.to_string(index=False))

# # Combine all results
# df_all_categories = pd.concat(all_category_results, ignore_index=True)
# df_all_summaries = pd.DataFrame(all_summaries)

# # Save results
# df_all_categories.to_csv("THRESHOLD_CATEGORY_RESULTS_BOTH_SOURCES.csv", index=False)
# df_all_summaries.to_csv("THRESHOLD_SUMMARY_BOTH_SOURCES.csv", index=False)

# print("\n" + "="*70)
# print("COMPLETE RESULTS SAVED")
# print("="*70)
# print("\n✅ THRESHOLD_CATEGORY_RESULTS_BOTH_SOURCES.csv")
# print("✅ THRESHOLD_SUMMARY_BOTH_SOURCES.csv")

# # ========================================================================
# # GENERATE COMPARISON TABLES
# # ========================================================================

# print("\n" + "="*70)
# print("GENERATING COMPARISON TABLES")
# print("="*70)

# for tau in thresholds:
#     print(f"\n{'='*70}")
#     print(f"COMPARISON AT τ = {tau}")
#     print(f"{'='*70}")
    
#     video_data = df_all_categories[
#         (df_all_categories['Source'] == 'VIDEO') & 
#         (df_all_categories['Threshold'] == tau)
#     ]
    
#     audio_data = df_all_categories[
#         (df_all_categories['Source'] == 'AUDIO') & 
#         (df_all_categories['Threshold'] == tau)
#     ]
    
#     # Create side-by-side comparison
#     comparison = video_data.merge(
#         audio_data,
#         on='Category',
#         suffixes=('_VIDEO', '_AUDIO')
#     )[['Category', 'Valence_Pct_VIDEO', 'Valence_Pct_AUDIO', 
#        'Arousal_Pct_VIDEO', 'Arousal_Pct_AUDIO']]
    
#     print(comparison.to_string(index=False))
    
#     # Save individual comparison
#     comparison.to_csv(f"COMPARISON_tau_{tau}.csv", index=False)
#     print(f"\n✅ Saved: COMPARISON_tau_{tau}.csv")

# print("\n" + "="*70)
# print("ALL ANALYSES COMPLETE!")
# print("="*70)
# print("\nGenerated 12 files:")
# print("  - THRESHOLD_CATEGORY_RESULTS_BOTH_SOURCES.csv (master file)")
# print("  - THRESHOLD_SUMMARY_BOTH_SOURCES.csv (agreement rates)")
# print("  - COMPARISON_tau_0.25.csv")
# print("  - COMPARISON_tau_0.5.csv")
# print("  - COMPARISON_tau_0.75.csv")
# print("  - COMPARISON_tau_1.0.csv")



"""
Complete Threshold Analysis - CORRECT VERSION
P = Perceived (what user thought clip showed)
I = Induced (what user actually felt)
E = Expressed (video/audio prediction)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

print("="*70)
print("THRESHOLD ANALYSIS - P.I.E. (CORRECT)")
print("="*70)

# ========================================================================
# LOAD DATA
# ========================================================================

print("\nLoading data...")

# Perceived (P) - What user thought clip was showing
df_perc_val = pd.read_csv("perceived_valence_summary.csv")
df_perc_aro = pd.read_csv("perceived_arousal_summary.csv")

# Induced (I) - What user actually felt
df_ind_val = pd.read_csv("induced_valence_summary.csv")
df_ind_aro = pd.read_csv("induced_arousal_summary.csv")

print(f"✅ Loaded perceived and induced emotions")

# ========================================================================
# CATEGORIZATION FUNCTION
# ========================================================================

def categorize_emotions(df_merged, tau):
    """
    Categorize emotions into 4 groups based on threshold
    P = Perceived, I = Induced, E = Expressed
    """
    df = df_merged.copy()
    
    # Calculate agreements
    df['PE_agree'] = abs(df['P'] - df['E']) <= tau
    df['PI_agree'] = abs(df['P'] - df['I']) <= tau
    df['IE_agree'] = abs(df['I'] - df['E']) <= tau
    
    # Categorize
    def get_category(row):
        if row['PE_agree'] and row['PI_agree']:
            return 'P=I=E'
        elif row['PE_agree'] and not row['PI_agree']:
            return 'P=E≠I'
        elif not row['PE_agree'] and row['PI_agree']:
            return 'P≠E=I'
        else:
            return 'P≠I≠E'
    
    df['category'] = df.apply(get_category, axis=1)
    
    return df

# ========================================================================
# ANALYSIS FUNCTION
# ========================================================================

def run_threshold_analysis(source_name, df_expr_val, df_expr_aro, tau):
    """
    Run analysis for one source at one threshold
    """
    
    # Merge VALENCE: P, I, E
    df_val = df_perc_val[['clip_id', 'avg_perceived_valence']].merge(
        df_ind_val[['clip_id', 'avg_induced_valence']],
        on='clip_id'
    ).merge(
        df_expr_val[['clip_id', 'expressed_valence']],
        on='clip_id'
    )
    df_val.columns = ['clip_id', 'P', 'I', 'E']
    
    # Merge AROUSAL: P, I, E
    df_aro = df_perc_aro[['clip_id', 'avg_perceived_arousal']].merge(
        df_ind_aro[['clip_id', 'avg_induced_arousal']],
        on='clip_id'
    ).merge(
        df_expr_aro[['clip_id', 'expressed_arousal']],
        on='clip_id'
    )
    df_aro.columns = ['clip_id', 'P', 'I', 'E']
    
    # Categorize
    df_val_cat = categorize_emotions(df_val, tau)
    df_aro_cat = categorize_emotions(df_aro, tau)
    
    # Calculate metrics for each category
    results = []
    
    for category in ['P=I=E', 'P=E≠I', 'P≠E=I', 'P≠I≠E']:
        # Valence
        val_subset = df_val_cat[df_val_cat['category'] == category]
        val_n = len(val_subset)
        val_pct = (val_n / len(df_val_cat) * 100) if len(df_val_cat) > 0 else 0
        
        # Arousal
        aro_subset = df_aro_cat[df_aro_cat['category'] == category]
        aro_n = len(aro_subset)
        aro_pct = (aro_n / len(df_aro_cat) * 100) if len(df_aro_cat) > 0 else 0
        
        results.append({
            'Source': source_name,
            'Threshold': tau,
            'Category': category,
            'Valence_N': val_n,
            'Valence_Pct': val_pct,
            'Arousal_N': aro_n,
            'Arousal_Pct': aro_pct
        })
    
    # Overall agreement rates
    val_pe_agree = df_val_cat['PE_agree'].sum() / len(df_val_cat) * 100
    aro_pe_agree = df_aro_cat['PE_agree'].sum() / len(df_aro_cat) * 100
    
    summary = {
        'Source': source_name,
        'Threshold': tau,
        'Valence_PE_Agreement': val_pe_agree,
        'Arousal_PE_Agreement': aro_pe_agree,
        'Total_Clips': len(df_val_cat)
    }
    
    return pd.DataFrame(results), summary

# ========================================================================
# RUN FOR ALL THRESHOLDS AND BOTH SOURCES
# ========================================================================

thresholds = [0.25, 0.5, 0.75, 1.0]

# Load VIDEO and AUDIO expressed emotions
df_video_val = pd.read_csv("final_expressed_valence_VIDEO.csv")
df_video_aro = pd.read_csv("final_expressed_arousal_VIDEO.csv")

df_audio_val = pd.read_csv("final_expressed_valence_AUDIO.csv")
df_audio_aro = pd.read_csv("final_expressed_arousal_AUDIO.csv")

all_category_results = []
all_summaries = []

print("\n" + "="*70)
print("RUNNING THRESHOLD ANALYSIS")
print("="*70)

for tau in thresholds:
    print(f"\n{'='*70}")
    print(f"THRESHOLD τ = {tau}")
    print(f"{'='*70}")
    
    # VIDEO source
    print(f"\n  VIDEO-expressed (τ={tau})...")
    video_results, video_summary = run_threshold_analysis(
        "VIDEO", df_video_val, df_video_aro, tau
    )
    all_category_results.append(video_results)
    all_summaries.append(video_summary)
    
    # AUDIO source
    print(f"  AUDIO-predicted (τ={tau})...")
    audio_results, audio_summary = run_threshold_analysis(
        "AUDIO", df_audio_val, df_audio_aro, tau
    )
    all_category_results.append(audio_results)
    all_summaries.append(audio_summary)
    
    # Show results for this threshold
    print(f"\n  VIDEO Results (τ={tau}):")
    print(video_results.to_string(index=False))
    
    print(f"\n  AUDIO Results (τ={tau}):")
    print(audio_results.to_string(index=False))

# Combine all results
df_all_categories = pd.concat(all_category_results, ignore_index=True)
df_all_summaries = pd.DataFrame(all_summaries)

# Save results
df_all_categories.to_csv("THRESHOLD_PIE_RESULTS_BOTH_SOURCES.csv", index=False)
df_all_summaries.to_csv("THRESHOLD_PIE_SUMMARY_BOTH_SOURCES.csv", index=False)

print("\n" + "="*70)
print("COMPLETE RESULTS SAVED")
print("="*70)
print("\n✅ THRESHOLD_PIE_RESULTS_BOTH_SOURCES.csv")
print("✅ THRESHOLD_PIE_SUMMARY_BOTH_SOURCES.csv")

# ========================================================================
# GENERATE COMPARISON TABLES
# ========================================================================

print("\n" + "="*70)
print("GENERATING COMPARISON TABLES")
print("="*70)

for tau in thresholds:
    print(f"\n{'='*70}")
    print(f"COMPARISON AT τ = {tau}")
    print(f"{'='*70}")
    
    video_data = df_all_categories[
        (df_all_categories['Source'] == 'VIDEO') & 
        (df_all_categories['Threshold'] == tau)
    ]
    
    audio_data = df_all_categories[
        (df_all_categories['Source'] == 'AUDIO') & 
        (df_all_categories['Threshold'] == tau)
    ]
    
    # Create side-by-side comparison
    comparison = video_data.merge(
        audio_data,
        on='Category',
        suffixes=('_VIDEO', '_AUDIO')
    )[['Category', 'Valence_Pct_VIDEO', 'Valence_Pct_AUDIO', 
       'Arousal_Pct_VIDEO', 'Arousal_Pct_AUDIO']]
    
    print(comparison.to_string(index=False))
    
    # Save individual comparison
    comparison.to_csv(f"COMPARISON_PIE_tau_{tau}.csv", index=False)
    print(f"\n✅ Saved: COMPARISON_PIE_tau_{tau}.csv")

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  - THRESHOLD_PIE_RESULTS_BOTH_SOURCES.csv")
print("  - THRESHOLD_PIE_SUMMARY_BOTH_SOURCES.csv")
print("  - COMPARISON_PIE_tau_0.25.csv through COMPARISON_PIE_tau_1.0.csv")