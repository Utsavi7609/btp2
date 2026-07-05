import pandas as pd

# 1. Load the input files
audio_df = pd.read_csv('true_music_voice_valence_final.csv')
video_df = pd.read_csv('Emotion_Analysis_Results_with_matches_video.csv')

# 2. Normalize Clip IDs (Remove .mp4 and whitespace to ensure matching)
audio_df['clip_id'] = audio_df['movie_clip'].str.replace('.mp4', '', case=False).str.strip()
video_df['clip_id'] = video_df['video_name'].str.replace('.mp4', '', case=False).str.strip()

# 3. Apply Audio Usefulness Rule
# Rule: If Music Probability >= 0.50 OR voice_ratio <= 0.10, it is non-useful (0).
# Else, it is useful (1).
def label_audio(row):
    # Handling potential missing values
    music_prob = row['Music Probability']
    voice_ratio = row['voice_ratio']
    if music_prob >= 0.50 or voice_ratio <= 0.10:
        return 0
    else:
        return 1

audio_df['audio_useful'] = audio_df.apply(label_audio, axis=1)

# 4. Apply Video Usefulness Logic
# Rule: Use the existing 'usefulness_tool' column. 'Useful' -> 1, 'Non-useful' -> 0.
video_df['video_useful'] = video_df['usefulness_tool'].apply(
    lambda x: 1 if str(x).strip().lower() == 'useful' else 0
)

# 5. Extract only necessary columns and Merge
audio_simplified = audio_df[['clip_id', 'audio_useful']].drop_duplicates()
video_simplified = video_df[['clip_id', 'video_useful']].drop_duplicates()

# We perform an outer merge to include clips present in either file
final_df = pd.merge(audio_simplified, video_simplified, on='clip_id', how='outer')

# Fill missing values with 0 (if a clip wasn't evaluated for one mode, assume non-useful)
final_df = final_df.fillna(0).astype({'audio_useful': int, 'video_useful': int})

# 6. Save the final output
final_df.to_csv('clip_usefulness_labels.csv', index=False)
print("Successfully generated 'clip_usefulness_labels.csv'")