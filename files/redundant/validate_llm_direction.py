import pandas as pd

def check_llm_direction(dim):
    # 1. Load Human Perception (What people saw)
    p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
    
    # 2. Load LLM Inference (What the body felt)
    i_df = pd.read_csv(f'inferred_{dim}_summary.csv')

    # Standardize IDs to ensure they match
    p_df['clip_id'] = p_df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
    i_df['clip_id'] = i_df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()

    # 3. Find the "Anchor" clips from Human Perception
    p_col = f'avg_{dim}'
    highest_p_clip = p_df.loc[p_df[p_col].idxmax()]
    lowest_p_clip = p_df.loc[p_df[p_col].idxmin()]

    print(f"\n==== {dim.upper()} DIRECTIONAL TEST ====")
    print(f"Happiest Clip (Perceived): {highest_p_clip['clip_id']} (Score: {highest_p_clip[p_col]})")
    print(f"Saddest Clip (Perceived):  {lowest_p_clip['clip_id']} (Score: {lowest_p_clip[p_col]})")

    # 4. Check these same clips in the LLM Inferred data
    i_col = f'avg_inferred_{dim}'
    
    # Get LLM scores for those specific clips
    hi_inferred = i_df[i_df['clip_id'] == highest_p_clip['clip_id']][i_col].values
    lo_inferred = i_df[i_df['clip_id'] == lowest_p_clip['clip_id']][i_col].values

    if len(hi_inferred) > 0 and len(lo_inferred) > 0:
        hi_val = hi_inferred[0]
        lo_val = lo_inferred[0]

        print("-" * 30)
        print(f"LLM Inferred Score for Happy Clip: {hi_val:.2f}")
        print(f"LLM Inferred Score for Sad Clip:   {lo_val:.2f}")

        # THE FINAL VERDICT
        if hi_val > lo_val:
            print("\nRESULT: SUCCESS! The LLM correctly ranked the Happy clip higher than the Sad clip.")
            print("The model is sensitive to heart rate changes. You can proceed to statistical validation.")
        else:
            print("\nRESULT: FAIL! The LLM ranked the Sad clip equal to or higher than the Happy clip.")
            print("ACTION: Your LLM is likely hallucinating. Check your prompt or heart rate data scaling.")
    else:
        print("\nERROR: Could not find the anchor clips in your inferred summary file.")

# Run for both Valence (Happy/Sad) and Arousal (Excited/Calm)
check_llm_direction('valence')
check_llm_direction('arousal')