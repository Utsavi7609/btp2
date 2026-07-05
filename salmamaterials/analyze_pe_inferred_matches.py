import pandas as pd
import os

def clean_id(series):
    """Ensures IDs match by removing extensions and whitespace."""
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def run_inferred_matching_analysis(dim):
    """
    dim: 'valence' or 'arousal'
    """
    print(f"\n--- Analyzing Inferred {dim.capitalize()} ---")
    
    # 1. LOAD THE THREE SOURCES
    # Perceived (p) - from your earlier summary
    p_file = f'perceived_{dim}_summary.csv'
    # Expressed (e) - from our ground truth script
    e_file = f'final_expressed_{dim}.csv'
    # Inferred Induced (i_inf) - from the LLM/Body signal summary
    i_inf_file = f'inferred_{dim}_summary.csv'

    # Check if files exist
    for f in [p_file, e_file, i_inf_file]:
        if not os.path.exists(f):
            print(f"ERROR: Missing {f}")
            return

    p_df = pd.read_csv(p_file)
    e_df = pd.read_csv(e_file)
    i_inf_df = pd.read_csv(i_inf_file)

    # 2. STANDARDIZE IDs
    p_df['clip_id'] = clean_id(p_df['clip_id'])
    e_df['clip_id'] = clean_id(e_df['clip_id'])
    i_inf_df['clip_id'] = clean_id(i_inf_df['clip_id'])

    # 3. MERGE (p and e)
    # Aligning perceived (p) and expressed (e) values
    # Column in perceived is 'avg_valence' or 'avg_arousal'
    p_col = f'avg_{dim}'
    pe_merged = pd.merge(
        p_df[['clip_id', p_col]], 
        e_df[['clip_id', 'expressed_value']], 
        on='clip_id'
    ).rename(columns={p_col: 'p_value', 'expressed_value': 'e_value'})

    # 4. FILTER 1: |p - e| <= 1 (Cognitive Success)
    pe_match = pe_merged[abs(pe_merged['p_value'] - pe_merged['e_value']) <= 1].copy()
    
    # Save the first set of files
    pe_filename = f'pe_inferred_match_{dim}.csv'
    pe_match.to_csv(pe_filename, index=False)
    print(f"1. PE Match: Found {len(pe_match)} clips where |p-e| <= 1. Saved to {pe_filename}")

    # 5. MERGE WITH INFERRED INDUCED (i_inf)
    # The column in your inferred summary is 'avg_inferred_valence' or 'avg_inferred_arousal'
    i_col = f'avg_inferred_{dim}'
    pei_inf_merged = pd.merge(
        pe_match, 
        i_inf_df[['clip_id', i_col]], 
        on='clip_id'
    ).rename(columns={i_col: 'i_inferred_value'})

    # 6. FILTER 2: |p - i_inferred| <= 1 (Physiological Resonance)
    pei_inf_match = pei_inf_merged[abs(pei_inf_merged['p_value'] - pei_inf_merged['i_inferred_value']) <= 1].copy()
    
    # Save the final resonance files
    pei_inf_filename = f'pei_inferred_match_{dim}.csv'
    pei_inf_match.to_csv(pei_inf_filename, index=False)
    print(f"2. PEI-Inferred Match: Of those, {len(pei_inf_match)} clips also satisfy |p-i_inf| <= 1. Saved to {pei_inf_filename}")

# --- EXECUTION ---
run_inferred_matching_analysis('valence')
run_inferred_matching_analysis('arousal')