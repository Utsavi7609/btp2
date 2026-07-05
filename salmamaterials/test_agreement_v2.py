import pandas as pd
import numpy as np

def calculate_clip_agreement():
    try:
        i_df = pd.read_csv('induced_valence_summary.csv')
        b_df = pd.read_csv('inferred_valence_summary_qwen_FS.csv')
        
        # Clip Level Agreement (N=274)
        # Assuming last column is the average
        i_avg = i_df[['clip_id', 'avg_valence']]
        b_avg = b_df[['clip_id', 'avg_inferred_valence']]
        
        merged = pd.merge(i_avg, b_avg, on='clip_id')
        
        thresholds = [0.25, 0.5, 0.75, 1.0]
        print(f"--- Clip-Level Agreement (N={len(merged)}) ---")
        for t in thresholds:
            matches = (np.abs(merged['avg_valence'] - merged['avg_inferred_valence']) <= t).sum()
            print(f"tau={t}: {matches} / {len(merged)} ({round(matches/len(merged)*100, 1)}%)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_clip_agreement()
