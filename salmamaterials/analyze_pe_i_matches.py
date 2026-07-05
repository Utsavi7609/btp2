import pandas as pd

def clean_id(series):
    """Ensures IDs match by removing extensions and whitespace."""
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def run_matching_analysis(dim):
    """
    dim: 'valence' or 'arousal'
    """
    print(f"\n--- Analyzing {dim.capitalize()} ---")
    
    # 1. LOAD THE THREE SOURCES
    # Perceived (p) - from your summary file
    p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
    # Expressed (e) - from the ground truth file we just made
    e_df = pd.read_csv(f'final_expressed_{dim}.csv')
    # Induced (i) - from your summary file
    i_df = pd.read_csv(f'induced_{dim}_summary.csv')

    # 2. STANDARDIZE IDs
    p_df['clip_id'] = clean_id(p_df['clip_id'])
    e_df['clip_id'] = clean_id(e_df['clip_id'])
    i_df['clip_id'] = clean_id(i_df['clip_id'])

    # 3. MERGE (p and e)
    # Aligning perceived (p) and expressed (e) values
    pe_merged = pd.merge(
        p_df[['clip_id', f'avg_{dim}']], 
        e_df[['clip_id', 'expressed_value']], 
        on='clip_id'
    ).rename(columns={f'avg_{dim}': 'p_value', 'expressed_value': 'e_value'})

    # 4. FILTER 1: |p - e| <= 1
    pe_match = pe_merged[abs(pe_merged['p_value'] - pe_merged['e_value']) <= 1].copy()
    
    # Save the first two files
    pe_filename = f'pe_match_{dim}.csv'
    pe_match.to_csv(pe_filename, index=False)
    print(f"1. PE Match: Found {len(pe_match)} clips where |p-e| <= 1. Saved to {pe_filename}")

    # 5. MERGE WITH INDUCED (i)
    # Aligning the matches with the induced (i) values
    pei_merged = pd.merge(
        pe_match, 
        i_df[['clip_id', f'avg_{dim}']], 
        on='clip_id'
    ).rename(columns={f'avg_{dim}': 'i_value'})

    # 6. FILTER 2: |p - i| <= 1 (Simultaneous Match)
    pei_match = pei_merged[abs(pei_merged['p_value'] - pei_merged['i_value']) <= 1].copy()
    
    # Save the next two files
    pei_filename = f'pei_match_{dim}.csv'
    pei_match.to_csv(pei_filename, index=False)
    print(f"2. PEI Match: Of those, {len(pei_match)} clips also satisfy |p-i| <= 1. Saved to {pei_filename}")

# --- EXECUTION ---
# This runs the logic for both Valence and Arousal
run_matching_analysis('valence')
run_matching_analysis('arousal')