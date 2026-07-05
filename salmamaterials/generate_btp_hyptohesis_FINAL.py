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
    return np.abs(a - b) <= t

def get_category_ids(p, e, i, t, clip_ids):
    mask_match = (is_m(p, e, t)) & (is_m(p, i, t))
    mask_feel  = (is_m(p, i, t)) & (~is_m(p, e, t))
    mask_drift = (is_m(p, e, t)) & (~is_m(p, i, t))
    
    match_ids = set(clip_ids[mask_match])
    feel_ids  = set(clip_ids[mask_feel])
    drift_ids = set(clip_ids[mask_drift])
    none_ids  = set(clip_ids) - match_ids - feel_ids - drift_ids
    
    return {'Match': match_ids, 'Feel': feel_ids, 'Drift': drift_ids, 'NoMatch': none_ids}

def run_final_btp_analysis():
    results = []

    for t in THRESHOLDS:
        print(f"Processing Threshold {t}...")
        for dim in DIMS:
            # Load Baselines
            try:
                p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
                i_df = pd.read_csv(f'induced_{dim}_summary.csv')
                e_vid = pd.read_csv(f'final_expressed_{dim}_VIDEO.csv')
                e_aud = pd.read_csv(f'final_expressed_{dim}_AUDIO.csv')
            except Exception as e:
                print(f"Error loading base files for {dim}: {e}")
                continue

            for df in [p_df, i_df, e_vid, e_aud]:
                df['clip_id'] = clean_id(df['clip_id'])

            # -- SELF ANALYSIS --
            m_v = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_vid[['clip_id', f'expressed_{dim}']], on='clip_id')
            m_v = pd.merge(m_v, i_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
            m_v.columns = ['id', 'p', 'e', 'i']
            v_cats = get_category_ids(m_v.p, m_v.e, m_v.i, t, m_v.id)
            
            m_a = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_aud[['clip_id', f'expressed_{dim}']], on='clip_id')
            m_a = pd.merge(m_a, i_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
            m_a.columns = ['id', 'p', 'e', 'i']
            a_cats = get_category_ids(m_a.p, m_a.e, m_a.i, t, m_a.id)
            
            for mod, cats in [('VIDEO', v_cats), ('AUDIO', a_cats)]:
                results.append({
                    'Threshold': t, 'Dimension': dim, 'Source': 'SELF', 'Modality': mod, 'Model': 'N/A',
                    'Match': len(cats['Match']), 'Feel': len(cats['Feel']), 'Drift': len(cats['Drift']), 'NoMatch': len(cats['NoMatch']), 'Total': len(m_a if mod=='AUDIO' else m_v)
                })
            
            intersect_ids = set(m_v.id) & set(m_a.id)
            results.append({
                'Threshold': t, 'Dimension': dim, 'Source': 'SELF', 'Modality': 'INTERSECT', 'Model': 'N/A',
                'Match': len(v_cats['Match'] & a_cats['Match']),
                'Feel': len(v_cats['Feel'] & a_cats['Feel']),
                'Drift': len(v_cats['Drift'] & a_cats['Drift']),
                'NoMatch': len(v_cats['NoMatch'] & a_cats['NoMatch']),
                'Total': len(intersect_ids)
            })

            # -- BODY ANALYSIS (MODELS) --
            for model in MODELS:
                try:
                    b_df = pd.read_csv(f'inferred_{dim}_summary_{model}_FS.csv')
                    b_df['clip_id'] = clean_id(b_df['clip_id'])
                    
                    mb_v = pd.merge(m_v[['id', 'p', 'e']], b_df[['clip_id', f'avg_inferred_{dim}']], left_on='id', right_on='clip_id')
                    vb_cats = get_category_ids(mb_v.p, mb_v.e, mb_v.iloc[:,-1], t, mb_v.id)
                    
                    mb_a = pd.merge(m_a[['id', 'p', 'e']], b_df[['clip_id', f'avg_inferred_{dim}']], left_on='id', right_on='clip_id')
                    ab_cats = get_category_ids(mb_a.p, mb_a.e, mb_a.iloc[:,-1], t, mb_a.id)
                    
                    for mod, cats in [('VIDEO', vb_cats), ('AUDIO', ab_cats)]:
                        results.append({
                            'Threshold': t, 'Dimension': dim, 'Source': 'BODY', 'Modality': mod, 'Model': model,
                            'Match': len(cats['Match']), 'Feel': len(cats['Feel']), 'Drift': len(cats['Drift']), 'NoMatch': len(cats['NoMatch']), 'Total': len(cats['Match'])+len(cats['Feel'])+len(cats['Drift'])+len(cats['NoMatch'])
                        })
                    
                    results.append({
                        'Threshold': t, 'Dimension': dim, 'Source': 'BODY', 'Modality': 'INTERSECT', 'Model': model,
                        'Match': len(vb_cats['Match'] & ab_cats['Match']),
                        'Feel': len(vb_cats['Feel'] & ab_cats['Feel']),
                        'Drift': len(vb_cats['Drift'] & ab_cats['Drift']),
                        'NoMatch': len(vb_cats['NoMatch'] & ab_cats['NoMatch']),
                        'Total': len(set(mb_v.id) & set(mb_a.id))
                    })
                except Exception as ex:
                    pass

    df_res = pd.DataFrame(results)
    df_res.to_csv('MASTER_BTP_HYPOTHESIS_V4_FINAL.csv', index=False)
    print("✅ Success! Generated MASTER_BTP_HYPOTHESIS_V4_FINAL.csv")
    
    # Validation Print
    for t in [0.25, 0.5, 0.75, 1.0]:
        print(f"\n--- TABLE FOR THRESHOLD {t} (Valence) ---")
        for mod in ['AUDIO', 'VIDEO', 'INTERSECT']:
            row = df_res[(df_res.Threshold==t) & (df_res.Source=='SELF') & (df_res.Modality==mod) & (df_res.Dimension=='valence')]
            if not row.empty:
                r = row.iloc[0]
                print(f" SELF | {mod:<9} | {r.Match:<4} | {r.Feel:<4} | {r.Drift:<4} | {r.NoMatch:<4} | Total: {r.Total}")

if __name__ == "__main__":
    run_final_btp_analysis()
