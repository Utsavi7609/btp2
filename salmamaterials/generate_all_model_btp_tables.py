import pandas as pd
import numpy as np
import os

# --- CONFIGURE MODELS & DATA ---
MODELS = ['llama', 'qwen', 'llama4']
MODALITIES = ['AUDIO', 'VIDEO']
THRESHOLDS = [0.25, 0.5, 0.75, 1.0]

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def is_m(a, b, t):
    return np.abs(a - b) <= t

def run_multi_model_analysis():
    all_rows = []

    for t in THRESHOLDS:
        print(f"\nProcessing Threshold: {t}...")
        for mod in MODALITIES:
            # Load Baselines
            for dim in ['valence', 'arousal']:
                p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
                e_df = pd.read_csv(f'final_expressed_{dim}_{mod}.csv')
                irep_df = pd.read_csv(f'induced_{dim}_summary.csv')

                # Standardize
                for df in [p_df, e_df, irep_df]:
                    df['clip_id'] = clean_id(df['clip_id'])

                # Base Merge for Self
                m_base = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', f'expressed_{dim}']], on='clip_id')
                m_base = pd.merge(m_base, irep_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_induced'))
                m_base.columns = ['id', 'p', 'e', 'i_self']

                # Self Counts
                s_h1 = ((is_m(m_base.p, m_base.e, t)) & (is_m(m_base.p, m_base.i_self, t))).sum()
                s_h2 = ((is_m(m_base.p, m_base.i_self, t)) & (~is_m(m_base.p, m_base.e, t))).sum()
                s_h3 = ((is_m(m_base.p, m_base.e, t)) & (~is_m(m_base.p, m_base.i_self, t))).sum()
                s_h4 = len(m_base) - (s_h1 + s_h2 + s_h3)

                # Store Self row (once per modality/dim/threshold)
                all_rows.append({
                    'Threshold': t, 'Modality': mod, 'Dimension': dim, 'Source': 'SELF', 'Model': 'N/A',
                    'All_Match': s_h1, 'Feel_Perception': s_h2, 'Understand': s_h3, 'No_Match': s_h4, 'Total': len(m_base)
                })

                # Now process each MODEL for Body
                for model in MODELS:
                    try:
                        ipred_df = pd.read_csv(f'inferred_{dim}_summary_{model}_FS.csv')
                        ipred_df['clip_id'] = clean_id(ipred_df['clip_id'])
                        
                        m_body = pd.merge(m_base, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
                        m_body.columns = ['id', 'p', 'e', 'i_self', 'i_body']

                        b_h1 = ((is_m(m_body.p, m_body.e, t)) & (is_m(m_body.p, m_body.i_body, t))).sum()
                        b_h2 = ((is_m(m_body.p, m_body.i_body, t)) & (~is_m(m_body.p, m_body.e, t))).sum()
                        b_h3 = ((is_m(m_body.p, m_body.e, t)) & (~is_m(m_body.p, m_body.i_body, t))).sum()
                        b_h4 = len(m_body) - (b_h1 + b_h2 + b_h3)

                        all_rows.append({
                            'Threshold': t, 'Modality': mod, 'Dimension': dim, 'Source': 'BODY', 'Model': model,
                            'All_Match': b_h1, 'Feel_Perception': b_h2, 'Understand': b_h3, 'No_Match': b_h4, 'Total': len(m_body)
                        })
                    except Exception as e:
                        print(f"  ⚠️ Error processing {model}/{mod}/{dim}: {e}")

    # Save to Master CSV
    df_final = pd.DataFrame(all_rows)
    df_final.to_csv('MASTER_BTP_MULTIMODEL_RESULTS.csv', index=False)
    print("\n✅ Success! Saved results to MASTER_BTP_MULTIMODEL_RESULTS.csv")

    # Final Summary for Llama-70B (User's primary model)
    print("\n--- SUMMARY FOR LLAMA-70B (VIDEO VALENCE) ---")
    for t in [0.25, 0.5, 0.75, 1.0]:
        sub_s = df_final[(df_final['Threshold']==t) & (df_final['Modality']=='VIDEO') & (df_final['Dimension']=='valence') & (df_final['Source']=='SELF')]
        sub_b = df_final[(df_final['Threshold']==t) & (df_final['Modality']=='VIDEO') & (df_final['Dimension']=='valence') & (df_final['Model']=='llama')]
        
        if not sub_s.empty and not sub_b.empty:
            print(f"| Threshold {t} | Self (V) | Body (V) |")
            print(f"| H1 (P=E=I)   | {sub_s.iloc[0]['All_Match']:<8} | {sub_b.iloc[0]['All_Match']:<8} |")
            print(f"| H2 (P=I)     | {sub_s.iloc[0]['Feel_Perception']:<8} | {sub_b.iloc[0]['Feel_Perception']:<8} |")
            print(f"| H3 (P=E)     | {sub_s.iloc[0]['Understand']:<8} | {sub_b.iloc[0]['Understand']:<8} |")
            print(f"| H4 (None)    | {sub_s.iloc[0]['No_Match']:<8} | {sub_b.iloc[0]['No_Match']:<8} |")
            print("-" * 35)

if __name__ == "__main__":
    run_multi_model_analysis()
