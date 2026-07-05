import pandas as pd

# Load your data
df = pd.read_csv('fitbit_hr_enhanced_context.csv')

def create_prompt(row):
    # This is the "Spoon-fed" question we ask the AI
    prompt = f"""
    Task: Predict Induced Emotion (1-5 scale)
    User Profile: This person usually feels what they see {row['V_Emotional_Resonance']}% of the time.
    Activity: Watching a clip from the movie {row['movie_name']}.
    Body Data: 
    - Average Heart Rate: {row['avg_bpm']} BPM
    - Heart Rate change: {row['bpm_drift']} BPM
    - Sequence: {row['hr_sequence']}
    
    Question: Based on the heart rate change, did they feel Valence (1=Sad, 5=Happy) and Arousal (1=Calm, 5=Excited)?
    Format: Return ONLY a JSON like this: {{"valence": X, "arousal": Y}}
    """
    return prompt.strip()

# Create the questions
df['llm_question'] = df.apply(create_prompt, axis=1)

# Save it
df.to_csv('fitbit_ready_for_llm.csv', index=False)
print("Done! Open 'fitbit_ready_for_llm.csv' to see the questions in the last column.")