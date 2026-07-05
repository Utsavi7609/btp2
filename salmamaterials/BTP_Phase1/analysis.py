import pandas as pd
import numpy as np

def run_analysis(filename, mode):
    df = pd.read_csv(filename)
    participants = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    
    # 1. CALCULATE EXPRESSED EMOTION (Consensus)
    # We take the median of all 'clip_valence/arousal' columns for each row
    perceived_cols = [f'{p}_clip_{mode}' for p in participants]
    df['Expressed_Consensus'] = df[perceived_cols].median(axis=1)

    results = []

    for p in participants:
        # Define column names for this participant
        p_initial = f'{p}_participant_{mode}'
        p_perceived = f'{p}_clip_{mode}'
        p_induced = f'{p}_impact_{mode}'
        
        # A. Expressed vs Perceived Match (Did they understand it?)
        # We check if their perception matches the consensus
        df[f'{p}_match_E_P'] = df[p_perceived] == df['Expressed_Consensus']
        
        # B. Perceived vs Induced Match (Did it work?)
        df[f'{p}_match_P_I'] = df[p_perceived] == df[p_induced]
        
        # C. Calculate Drift (Induced - Initial)
        df[f'{p}_drift'] = df[p_induced] - df[p_initial]

        # Aggregate stats for this user
        ep_match_count = df[f'{p}_match_E_P'].sum()
        pi_match_count = df[f'{p}_match_P_I'].sum()
        avg_drift = df[f'{p}_drift'].mean()

        results.append({
            'Participant': p,
            'E-P Matches (Understanding)': ep_match_count,
            'P-I Matches (Feeling)': pi_match_count,
            'Avg Drift': round(avg_drift, 3)
        })

    # Save the detailed matching data back to a new CSV
    output_detail = f'detailed_matching_{mode}.csv'
    df.to_csv(output_detail, index=False)
    
    return pd.DataFrame(results), output_detail

# Execute for Valence
valence_stats, v_file = run_analysis('valence_btp2.csv', 'valence')
# Execute for Arousal
arousal_stats, a_file = run_analysis('arousal_btp2.csv', 'arousal')

print("--- VALENCE MATCHING SUMMARY ---")
print(valence_stats)
print(f"\nDetailed file saved as: {v_file}")

print("\n--- AROUSAL MATCHING SUMMARY ---")
print(arousal_stats)
print(f"\nDetailed file saved as: {a_file}")