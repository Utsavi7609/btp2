import pandas as pd

def chain_emotions(file_name, type_name):
    df = pd.read_csv(file_name)
    
    # Sort by clip_order to ensure we are following the actual sequence
    df = df.sort_values('clip_order').reset_index(drop=True)
    
    participants = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    
    for p in participants:
        part_col = f'{p}_participant_{type_name}'
        impact_col = f'{p}_impact_{type_name}'
        
        # We start from the second clip (index 1)
        for i in range(1, len(df)):
            # Set the INITIAL of the current clip to the INDUCED (Impact) of the previous clip
            df.loc[i, part_col] = df.loc[i-1, impact_col]
            
    # Save the updated "Chained" version
    output_name = f'{type_name}_btp2_chained.csv'
    df.to_csv(output_name, index=False)
    print(f"Success! {output_name} generated with chained emotional drift.")

# Run for both Valence and Arousal
chain_emotions('valence_btp2.csv', 'valence')
chain_emotions('arousal_btp2.csv', 'arousal') # Uncomment if you have the arousal file ready