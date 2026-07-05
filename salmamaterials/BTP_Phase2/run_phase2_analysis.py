import pandas as pd
import numpy as np
import os

def perform_complete_analysis(filename, mode):
    """
    Performs the full statistical analysis for a given emotion dimension (Valence or Arousal).
    """
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Please run Phase 1 first.")
        return None, None

    # Load data
    df = pd.read_csv(filename)
    participants = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    
    # 1. Calculate Group Consensus (Median of what people perceived)
    perceived_cols = [f'{p}_clip_{mode}' for p in participants]
    df['Consensus_Perception'] = df[perceived_cols].median(axis=1)

    # 2. Participant Analysis (How each person performed)
    participant_stats = []
    for p in participants:
        p_init = f'{p}_participant_{mode}'
        p_perc = f'{p}_clip_{mode}'
        p_ind = f'{p}_impact_{mode}'
        
        # Calculation: How often user perception matches group consensus
        cog_align = (df[p_perc] == df['Consensus_Perception']).mean() * 100
        # Calculation: How often user feels what they recognize
        emo_res = (df[p_perc] == df[p_ind]).mean() * 100
        # Calculation: Average numerical change from start to end
        avg_drift = (df[p_ind] - df[p_init]).mean()
        
        participant_stats.append({
            'Participant': p,
            'Cognitive_Alignment_Percentage': round(cog_align, 2),
            'Emotional_Resonance_Percentage': round(emo_res, 2),
            'Average_Drift': round(avg_drift, 3)
        })

    # 3. Full Clip Ranking (Analysis of every single video)
    clip_analysis = []
    for idx, row in df.iterrows():
        # How many participants' felt what they perceived (Resonance)
        res_matches = [row[f'{p}_clip_{mode}'] == row[f'{p}_impact_{mode}'] for p in participants]
        # Numerical change magnitude across all participants
        drifts = [row[f'{p}_impact_{mode}'] - row[f'{p}_participant_{mode}'] for p in participants]
        
        clip_analysis.append({
            'clip_title': row['clip_title'],
            'Resonance_Rate': round(np.mean(res_matches), 3),
            'Average_Drift': round(np.mean(drifts), 3),
            'Absolute_Drift_Magnitude': round(abs(np.mean(drifts)), 3)
        })

    participant_df = pd.DataFrame(participant_stats)
    # Sort clips by the magnitude of drift (Highest impact first)
    clip_df = pd.DataFrame(clip_analysis).sort_values(by='Absolute_Drift_Magnitude', ascending=False)
    
    return participant_df, clip_df

# --- EXECUTION ---

print("Starting Phase 2 Analysis...")

# Process both dimensions
v_participants, v_clips = perform_complete_analysis('valence_btp2.csv', 'valence')
a_participants, a_clips = perform_complete_analysis('arousal_btp2.csv', 'arousal')

# Save all CSV files
v_participants.to_csv('participant_analysis_valence.csv', index=False)
a_participants.to_csv('participant_analysis_arousal.csv', index=False)
v_clips.to_csv('clip_drift_ranking_valence.csv', index=False)
a_clips.to_csv('clip_drift_ranking_arousal.csv', index=False)

# Generate the Professional Text Report
report_text = f"""BTP-2 PHASE 2: EMOTIONAL DRIFT AND CONSISTENCY ANALYSIS

DEFINITION OF METRICS:
1. Cognitive Alignment Percentage: This shows how often a participant's perception 
   of a clip's emotion matched the overall group consensus (the median).
2. Emotional Resonance Percentage: This shows how often a participant actually felt 
   the emotion they recognized in the movie clip.
3. Average Drift: This is the average numerical change between a participant's 
   starting emotion and their final emotion after watching the clips.
4. Resonance Rate: The proportion of participants (from 0.0 to 1.0) who felt 
   the emotion they perceived in a specific clip.
5. Absolute Drift Magnitude: The total degree of emotional change caused by a clip, 
   regardless of whether it moved the user up or down the scale.

SECTION 1: VALENCE ANALYSIS (Positive/Negative Emotion)
------------------------------------------------------
PARTICIPANT SUMMARY:
{v_participants.to_string(index=False)}

TOP 10 HIGHEST IMPACT CLIPS (Drift):
{v_clips[['clip_title', 'Average_Drift', 'Absolute_Drift_Magnitude']].head(10).to_string(index=False)}

TOP 10 MOST STABLE CLIPS (Consistent):
{v_clips[v_clips['Resonance_Rate'] >= 0.6].sort_values(by='Absolute_Drift_Magnitude').head(10)[['clip_title', 'Average_Drift', 'Absolute_Drift_Magnitude']].to_string(index=False)}


SECTION 2: AROUSAL ANALYSIS (Intensity/Energy)
------------------------------------------------------
PARTICIPANT SUMMARY:
{a_participants.to_string(index=False)}

TOP 10 HIGHEST IMPACT CLIPS (Drift):
{a_clips[['clip_title', 'Average_Drift', 'Absolute_Drift_Magnitude']].head(10).to_string(index=False)}

TOP 10 MOST STABLE CLIPS (Consistent):
{a_clips[a_clips['Resonance_Rate'] >= 0.6].sort_values(by='Absolute_Drift_Magnitude').head(10)[['clip_title', 'Average_Drift', 'Absolute_Drift_Magnitude']].to_string(index=False)}

Note: Full sorted lists for all 275 clips are saved in the CSV files.
"""

with open('BTP2_Phase2_Final_Report.txt', 'w') as f:
    f.write(report_text)

print("\nSUCCESS: All files generated.")
print("- participant_analysis_valence/arousal.csv")
print("- clip_drift_ranking_valence/arousal.csv")
print("- BTP2_Phase2_Final_Report.txt")