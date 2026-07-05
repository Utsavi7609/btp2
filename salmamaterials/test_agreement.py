import pandas as pd
import numpy as np

def calculate_detailed_agreement():
    try:
        # Load the data
        i_df = pd.read_csv('induced_valence_summary.csv')
        b_df = pd.read_csv('inferred_valence_summary_qwen_FS.csv')
        
        def melt_df(df, val_name):
            cols = [c for c in df.columns if c != 'clip_id']
            return df.melt(id_vars='clip_id', value_vars=cols, var_name='user', value_name=val_name)
            
        i_melt = melt_df(i_df, 'Induced')
        b_melt = melt_df(b_df, 'Inferred')
        
        merged = pd.merge(i_melt, b_melt, on=['clip_id', 'user']).dropna()
        
        thresholds = [0.25, 0.5, 0.75, 1.0]
        print(f"--- Response-Level Agreement (N={len(merged)}) ---")
        for t in thresholds:
            matches = (np.abs(merged['Induced'] - merged['Inferred']) <= t).sum()
            print(f"tau={t}: {matches} / {len(merged)} ({round(matches/len(merged)*100, 1)}%)")
            
        i_avg = i_melt.groupby('clip_id')['Induced'].mean().reset_index()
        b_avg = b_melt.groupby('clip_id')['Inferred'].mean().reset_index()
        merged_avg = pd.merge(i_avg, b_avg, on='clip_id')
        
        print(f"\n--- Clip-Level Agreement (N={len(merged_avg)}) ---")
        for t in thresholds:
            matches = (np.abs(merged_avg['Induced'] - merged_avg['Inferred']) <= t).sum()
            print(f"tau={t}: {matches} / {len(merged_avg)} ({round(matches/len(merged_avg)*100, 2)}%)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_detailed_agreement()
