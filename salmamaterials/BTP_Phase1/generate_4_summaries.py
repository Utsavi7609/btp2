import pandas as pd

# List of participants
participants = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']

def generate_report(input_file, score_type, dim, output_name):
    """
    input_file: 'valence_btp2.csv' or 'arousal_btp2.csv'
    score_type: 'clip' (perceived) or 'impact' (induced)
    dim: 'valence' or 'arousal'
    """
    try:
        df = pd.read_csv(input_file)
        
        # 1. Start with the Clip ID
        result = df[['clip_title']].rename(columns={'clip_title': 'clip_id'})
        
        # 2. Extract the 6 User columns for this specific dimension
        score_cols = []
        for p in participants:
            col_name = f"{p}_{score_type}_{dim}"
            if col_name in df.columns:
                result[f"{p}_{dim}"] = df[col_name]
                score_cols.append(col_name)
            else:
                result[f"{p}_{dim}"] = None
        
        # 3. Calculate Average (Truncated to 3 decimal places)
        # We use round(3) to keep it as a float while limiting the digits
        result[f'avg_{dim}'] = df[score_cols].mean(axis=1).round(3)
        
        # 4. Save to CSV
        result.to_csv(output_name, index=False)
        print(f"Successfully created: {output_name}")
        
    except FileNotFoundError:
        print(f"Skipping: {input_file} not found in this folder.")

# --- RUNNING THE GENERATION ---

# 1. Perceived Valence
generate_report('valence_btp2.csv', 'clip', 'valence', 'perceived_valence_summary.csv')

# 2. Induced Valence
generate_report('valence_btp2.csv', 'impact', 'valence', 'induced_valence_summary.csv')

# 3. Perceived Arousal
generate_report('arousal_btp2.csv', 'clip', 'arousal', 'perceived_arousal_summary.csv')

# 4. Induced Arousal
generate_report('arousal_btp2.csv', 'impact', 'arousal', 'induced_arousal_summary.csv')