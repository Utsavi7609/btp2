import pandas as pd
import os

def run_prediction_analysis(label, dim, p_file, e_file, irep_file, ipred_file, irep_col, ipred_col):
    def load(f):
        if not os.path.exists(f):
            print(f"Warning: File {f} not found.")
            return None
        df = pd.read_csv(f)
        df['clip_id'] = df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
        return df

    # Load all 4 data sources
    p_df = load(p_file)
    e_df = load(e_file)
    irep_df = load(irep_file)
    ipred_df = load(ipred_file)
    
    # FIXED: Proper check for missing dataframes
    if p_df is None or e_df is None or irep_df is None or ipred_df is None:
        return
    
    # Merge into a single master table for this dimension
    m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
    m = pd.merge(m, irep_df[['clip_id', irep_col]], on='clip_id')
    m = pd.merge(m, ipred_df[['clip_id', ipred_col]], on='clip_id')
    m.columns = ['id', 'p', 'e', 'irep', 'ipred']

    def match(a, b): return abs(a - b) <= 1

    stats = {
        "TOTAL CLIPS ANALYZED": len(m),
        "Q1: Perfect Sync (Reported == Predicted)": 0,
        "Q2: The Mind-Body Gap (Reported != Predicted)": 0,
        "  - Case A: Body matches Perception (ipred=p)": 0,
        "  - Case B: Body matches Ground Truth (ipred=e)": 0,
        "  - Case C: Unique Bio-Response (matches neither p nor e)": 0,
        "EXTRA: Perfect Predicted Alignment (p=e=ipred)": 0
    }

    for _, row in m.iterrows():
        sync = match(row['irep'], row['ipred'])
        
        # Perfect Alignment check for Predicted (ipred)
        if match(row['p'], row['e']) and match(row['p'], row['ipred']):
            stats["EXTRA: Perfect Predicted Alignment (p=e=ipred)"] += 1

        if sync:
            stats["Q1: Perfect Sync (Reported == Predicted)"] += 1
        else:
            stats["Q2: The Mind-Body Gap (Reported != Predicted)"] += 1
            matches_p = match(row['ipred'], row['p'])
            matches_e = match(row['ipred'], row['e'])
            
            # Categorize the Gap cases
            if matches_p and matches_e:
                # If it matches both, we prioritize Ground Truth as "Correct"
                stats["  - Case B: Body matches Ground Truth (ipred=e)"] += 1
            elif matches_p:
                stats["  - Case A: Body matches Perception (ipred=p)"] += 1
            elif matches_e:
                stats["  - Case B: Body matches Ground Truth (ipred=e)"] += 1
            else:
                stats["  - Case C: Unique Bio-Response (matches neither p nor e)"] += 1

    print(f"\n{label}")
    print("="*50)
    for k, v in stats.items():
        print(f"{k:45}: {v}")

if __name__ == "__main__":
    # Run for VALENCE
    run_prediction_analysis(
        "VALENCE: Mind (Reported) vs Body (Predicted)", 'valence', 
        'perceived_valence_summary.csv', 'final_expressed_valence.csv', 
        'induced_valence_summary.csv', 'inferred_valence_summary.csv',
        'avg_valence', 'avg_inferred_valence'
    )
    
    # Run for AROUSAL
    run_prediction_analysis(
        "AROUSAL: Mind (Reported) vs Body (Predicted)", 'arousal', 
        'perceived_arousal_summary.csv', 'final_expressed_arousal.csv', 
        'induced_arousal_summary.csv', 'inferred_arousal_summary.csv',
        'avg_arousal', 'avg_inferred_arousal'
    )