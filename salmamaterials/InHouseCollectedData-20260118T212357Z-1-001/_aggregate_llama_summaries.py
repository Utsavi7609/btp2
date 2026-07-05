import pandas as pd
import os

OUTPUT_DIR = r"d:\BTP\btp2\salmamaterials"
INPUT_CSV = os.path.join(OUTPUT_DIR, "InHouseCollectedData-20260118T212357Z-1-001", "_inferred_clip_emotions_llama.csv")

def aggregate_summaries():
    print("Loading raw inference data...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}. Have you finished running the inference script?")
        return

    # Clean clip titles (just in case they have .mp4 extensions)
    df['clip_id'] = df['clip_title'].str.replace('.mp4', '', regex=False)

    print(f"Found {len(df)} rows. Pivoting for Valence...")
    
    # Pivot for Valence
    valence_df = df.pivot_table(index='clip_id', columns='participant', values='inferred_valence', aggfunc='first').reset_index()
    # Ensure all users exist even if missing data
    users = ['Nishant', 'ajaycc17', 'aritra', 'debjit2001', 'pranjal123', 'sayantankuila']
    for u in users:
        if u not in valence_df.columns:
            valence_df[u] = None
            
    val_cols = ['clip_id'] + users
    valence_df = valence_df[val_cols]
    valence_df['avg_inferred_valence'] = valence_df[users].mean(axis=1).round(2)
    
    val_out = os.path.join(OUTPUT_DIR, "_inferred_valence_summary_llama.csv")
    valence_df.to_csv(val_out, index=False)
    print(f"Saved Valence summary to: {val_out}")

    print("Pivoting for Arousal...")
    # Pivot for Arousal
    arousal_df = df.pivot_table(index='clip_id', columns='participant', values='inferred_arousal', aggfunc='first').reset_index()
    for u in users:
        if u not in arousal_df.columns:
            arousal_df[u] = None
            
    aro_cols = ['clip_id'] + users
    arousal_df = arousal_df[aro_cols]
    arousal_df['avg_inferred_arousal'] = arousal_df[users].mean(axis=1).round(2)
    
    aro_out = os.path.join(OUTPUT_DIR, "_inferred_arousal_summary_llama.csv")
    arousal_df.to_csv(aro_out, index=False)
    print(f"Saved Arousal summary to: {aro_out}")

if __name__ == "__main__":
    aggregate_summaries()
