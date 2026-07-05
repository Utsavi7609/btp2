import pandas as pd
import os

def clean_id(series):
    """Removes file extensions and spaces for clean matching."""
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# 1. Load the LLM Results
input_file = 'llm_results_final2.csv'
if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please ensure the file is in this folder.")
else:
    df = pd.read_csv(input_file)
    df['clip_id'] = clean_id(df['clip_title'])

    # 2. Convert all emotion scores to numeric (handling any string/text errors)
    cols_to_fix = [
        'self_reported_valence', 'self_reported_arousal', 
        'inferred_valence', 'inferred_arousal'
    ]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Calculate the Average for each Clip across all available participants
    # This gives us one 'Reported' score and one 'Inferred' score per clip
    results = df.groupby('clip_id').agg({
        'self_reported_valence': 'mean',
        'self_reported_arousal': 'mean',
        'inferred_valence': 'mean',
        'inferred_arousal': 'mean'
    }).rename(columns={
        'self_reported_valence': 'avg_reported_v',
        'self_reported_arousal': 'avg_reported_a',
        'inferred_valence': 'avg_inferred_v',
        'inferred_arousal': 'avg_inferred_a'
    })

    # 4. Define the Match Condition (|Difference| <= 1.0)
    # We check if both Valence and Arousal are within the 1-point threshold
    v_match = abs(results['avg_reported_v'] - results['avg_inferred_v']) <= 1
    a_match = abs(results['avg_reported_a'] - results['avg_inferred_a']) <= 1

    # Filter for clips that meet BOTH conditions
    matching_clips = results[v_match & a_match].copy()

    # 5. Save only the matching clips to a new CSV
    output_file = 'reported_inferred_matching_clips.csv'
    matching_clips.to_csv(output_file)

    print("-" * 30)
    print(f"Analysis Complete!")
    print(f"Total Unique Clips Analyzed: {len(results)}")
    print(f"Total Matching Clips Found: {len(matching_clips)}")
    print(f"Results saved to: {output_file}")
    print("-" * 30)