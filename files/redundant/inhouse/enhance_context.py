import pandas as pd

# 1. Load the files
fitbit_df = pd.read_csv('fitbit_hr_context.csv')
v_part_df = pd.read_csv('participant_analysis_valence.csv')
a_part_df = pd.read_csv('participant_analysis_arousal.csv')

# 2. Rename columns for clarity before merging
v_part_df = v_part_df.rename(columns={
    'Cognitive_Alignment_Percentage': 'V_Cognitive_Alignment',
    'Emotional_Resonance_Percentage': 'V_Emotional_Resonance',
    'Average_Drift': 'V_Avg_Drift'
})
a_part_df = a_part_df.rename(columns={
    'Cognitive_Alignment_Percentage': 'A_Cognitive_Alignment',
    'Emotional_Resonance_Percentage': 'A_Emotional_Resonance',
    'Average_Drift': 'A_Avg_Drift'
})

# 3. Merge Participant Stats (Context Enhancement)
enhanced_df = fitbit_df.merge(v_part_df, left_on='participant', right_on='Participant', how='left')
enhanced_df = enhanced_df.merge(a_part_df, left_on='participant', right_on='Participant', how='left')

# Drop the duplicate Participant columns
enhanced_df = enhanced_df.drop(columns=['Participant_x', 'Participant_y'])

# 4. Extract Movie Name (Placeholder for Phase 4 Genre)
# This allows us to group results by movie until we have the specific clip genre
enhanced_df['movie_name'] = enhanced_df['clip_title'].apply(lambda x: x.split('_')[0])

# 5. Save the final "LLM-Ready" dataset
enhanced_df.to_csv('fitbit_hr_enhanced_context.csv', index=False)

print("SUCCESS: 'fitbit_hr_enhanced_context.csv' generated.")
print("This file now contains Heart Rate + User Emotional Profiles + Movie Identifiers.")