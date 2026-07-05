import pandas as pd
import numpy as np
import os

# --- CONFIGURE ---
MODELS = ['llama', 'qwen', 'llama4']
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]
DIMS = ['valence', 'arousal']

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def is_m(a, b, t):
    """Check if variable A and B are within tolerance Threshold."""
    return np.abs(a - b) <= t

def get_5_categories(p, e, i, t, clip_ids):
    """
    Splits the data into 5 scientific buckets:
    H1: Match (P=E=I)
    H2: Feel (P=I != E)
    H3: Drift (P=E != I)
    H4: Cognitive Dissonance (E=I != P)
    H5: Absolute Chaos (None of the above)
    """
    # Precision masks
    m_pe = is_m(p, e, t)
    m_pi = is_m(p, i, t)
    m_ei = is_m(e, i, t)

    mask_h1 = m_pe & m_pi
    mask_h2 = m_pi & (~m_pe)
    mask_h3 = m_pe & (~m_pi)
    mask_h4 = m_ei & (~m_pe) & (~m_pi) # E=I but P is the odd one out
    
    h1_ids = set(clip_ids[mask_h1])
    h2_ids = set(clip_ids[mask_h2])
    h3_ids = set(clip_ids[mask_h3])
    h4_ids = set(clip_ids[mask_h4])
    h5_ids = set(clip_ids) - h1_ids - h2_ids - h3_ids - h4_ids
    
    return {'H1': h1_ids, 'H2': h2_ids, 'H3': h3_ids, 'H4': h4_ids, 'H5': h5_ids}

def run_v5_analysis():
    results_buckets = []
    results_agreement = [] # For Task 3: I = B

    for t in THRESHOLDS:
        print(f"Processing Threshold {t}...")
        for dim in DIMS:
            # Load Baselines
            try:
                p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
                i_df = pd.read_csv(f'induced_{dim}_summary.csv')
                e_vid = pd.read_csv(f'final_expressed_{dim}_VIDEO.csv')
                e_aud = pd.read_csv(f'final_expressed_{dim}_AUDIO.csv')
            except: continue

            for df in [p_df, i_df, e_vid, e_aud]:
                df['clip_id'] = clean_id(df['clip_id'])

            # -- SELF ANALYSIS (Merging P, E, I) --
            m_v = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_vid[['clip_id', f'expressed_{dim}']], on='clip_id')
            m_v = pd.merge(m_v, i_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
            m_v.columns = ['id', 'p', 'e', 'i']
            
            m_a = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_aud[['clip_id', f'expressed_{dim}']], on='clip_id')
            m_a = pd.merge(m_a, i_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
            m_a.columns = ['id', 'p', 'e', 'i']

            # SELF BUCKETS
            for mod, df_mod in [('VIDEO', m_v), ('AUDIO', m_a)]:
                cats = get_5_categories(df_mod.p, df_mod.e, df_mod.i, t, df_mod.id)
                results_buckets.append({
                    'Threshold': t, 'Dimension': dim, 'Source': 'SELF', 'Modality': mod,
                    'H1_Match': len(cats['H1']), 'H2_Feel': len(cats['H2']), 'H3_Drift': len(cats['H3']), 
                    'H4_CBT': len(cats['H4']), 'H5_Chaos': len(cats['H5']), 'Total': len(df_mod)
                })

            # -- BODY ANALYSIS (Replacing E with B) --
            for model in MODELS:
                try:
                    summary_file = f'inferred_{dim}_summary_{model}_FS.csv'
                    if not os.path.exists(summary_file):
                        print(f"Skipping {model}: {summary_file} not found.")
                        continue
                    
                    b_df = pd.read_csv(summary_file)
                    b_df['clip_id'] = clean_id(b_df['clip_id'])
                    
                    # Merge P, B, E (Task 4 Confirmation)
                    mb_v = pd.merge(m_v[['id', 'p', 'e', 'i']], b_df[['clip_id', f'avg_inferred_{dim}']], left_on='id', right_on='clip_id').drop(columns=['clip_id'])
                    mb_v.columns = ['id', 'p', 'e', 'i', 'b'] # Using 'i' only for Task 3 sync
                    
                    mb_a = pd.merge(m_a[['id', 'p', 'e', 'i']], b_df[['clip_id', f'avg_inferred_{dim}']], left_on='id', right_on='clip_id').drop(columns=['clip_id'])
                    mb_a.columns = ['id', 'p', 'e', 'i', 'b']

                    print(f"Model {model} ({mod}): Merged rows = {len(mb_v)} (Video), {len(mb_a)} (Audio)")

                    # INTERSECT Calculation (Where Video and Audio Expressions agree)
                    # We need the clips where Video E = Audio E
                    # Then we categorize for those clips
                    shared_ids = set(mb_v.id).intersection(set(mb_a.id))
                    mb_v_shared = mb_v[mb_v.id.isin(shared_ids)].sort_values('id')
                    mb_a_shared = mb_a[mb_a.id.isin(shared_ids)].sort_values('id')
                    
                    # Intersect E means mb_v.e == mb_a.e
                    intersect_mask = (mb_v_shared.e == mb_a_shared.e)
                    mb_intersect = mb_v_shared[intersect_mask].copy()

                    # BUCKETS (Comparing P, B, E)
                    for mod, df_mb in [('VIDEO', mb_v), ('AUDIO', mb_a), ('INTERSECT', mb_intersect)]:
                        cats = get_5_categories(df_mb.p, df_mb.b, df_mb.e, t, df_mb.id)
                        results_buckets.append({
                            'Threshold': t, 'Dimension': dim, 'Source': 'BODY', 'Modality': mod, 'Model': model,
                            'H1_Match': len(cats['H1']), 'H2_Feel': len(cats['H2']), 'H3_Drift': len(cats['H3']), 
                            'H4_CBT': len(cats['H4']), 'H5_Chaos': len(cats['H5']), 'Total': len(df_mb)
                        })
                        
                        # TASK 3 & 5: Universal Truth Table (Clips where I = B)
                        if mod != 'INTERSECT':
                            # Calculation: abs(i - b) <= t
                            agree_mask = is_m(df_mb.i, df_mb.b, t)
                            df_truth = df_mb[agree_mask].copy()
                            
                            # Categorize P, E, and Agreement State (using b as proxy for i)
                            truth_cats = get_5_categories(df_truth.p, df_truth.b, df_truth.e, t, df_truth.id)
                            results_buckets.append({
                                'Threshold': t, 'Dimension': dim, 'Source': 'TRUTH', 'Modality': mod, 'Model': model,
                                'H1_Match': len(truth_cats['H1']), 'H2_Feel': len(truth_cats['H2']), 'H3_Drift': len(truth_cats['H3']), 
                                'H4_CBT': len(truth_cats['H4']), 'H5_Chaos': len(truth_cats['H5']), 'Total': len(df_truth)
                            })
                            
                            # Summary Stats
                            agree_size = len(df_truth)
                            results_agreement.append({
                                'Threshold': t, 'Dimension': dim, 'Modality': mod, 'Model': model,
                                'Agreed_Clips': agree_size, 'Total_Clips': len(df_mb),
                                'Agreement_Rate': round(agree_size/len(df_mb)*100, 1) if len(df_mb)>0 else 0
                            })
                        else:
                            # Intersect for TRUTH
                            # This means (Video I=B) AND (Audio I=B) AND (Video E = Audio E)
                            # To simplify, we can just look at the mb_intersect clips where I=B
                            agree_mask = is_m(df_mb.i, df_mb.b, t)
                            df_truth = df_mb[agree_mask].copy()
                            truth_cats = get_5_categories(df_truth.p, df_truth.b, df_truth.e, t, df_truth.id)
                            results_buckets.append({
                                'Threshold': t, 'Dimension': dim, 'Source': 'TRUTH', 'Modality': mod, 'Model': model,
                                'H1_Match': len(truth_cats['H1']), 'H2_Feel': len(truth_cats['H2']), 'H3_Drift': len(truth_cats['H3']), 
                                'H4_CBT': len(truth_cats['H4']), 'H5_Chaos': len(truth_cats['H5']), 'Total': len(df_truth)
                            })
                except Exception as e:
                    print(f"Error processing {model}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Save Master Results
    pd.DataFrame(results_buckets).to_csv('MASTER_BTP_HYPOTHESIS_V5_5BUCKET.csv', index=False)
    pd.DataFrame(results_agreement).to_csv('MIND_BODY_AGREEMENT_TABLE.csv', index=False)
    print("✅ Success! Generated 5-Bucket Matrix and Agreement Table.")

if __name__ == "__main__":
    run_v5_analysis()
