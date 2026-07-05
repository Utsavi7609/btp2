import pandas as pd
import numpy as np

def generate_report(filename, mode):
    df = pd.read_csv(filename)
    participants = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    
    # 1. Consensus & Resonance Logic
    perceived_cols = [f'{p}_clip_{mode}' for p in participants]
    df['Expressed_Consensus'] = df[perceived_cols].median(axis=1)

    user_data = []
    clip_summaries = []

    # 2. Analyze each Participant
    for p in participants:
        p_init, p_perc, p_ind = f'{p}_participant_{mode}', f'{p}_clip_{mode}', f'{p}_impact_{mode}'
        
        cog_align = (df[p_perc] == df['Expressed_Consensus']).mean() * 100
        emo_res = (df[p_perc] == df[p_ind]).mean() * 100
        avg_drift = (df[p_ind] - df[p_init]).mean()
        
        # Determine Profile
        profile = "Typical Viewer" if cog_align > 35 else "Unique Interpreter"
        resonance = "High Resonance (Consistent)" if emo_res > 60 else "Low Resonance (Inconsistent)"
        
        user_data.append({
            'User': p,
            'Cognitive_Alignment_%': round(cog_align, 1),
            'Emotional_Resonance_%': round(emo_res, 1),
            'Avg_Drift': round(avg_drift, 2),
            'Profile': f"{profile} | {resonance}"
        })

    # 3. Rank the Clips (The "Shouting" Part)
    for idx, row in df.iterrows():
        res_matches = [row[f'{p}_clip_{mode}'] == row[f'{p}_impact_{mode}'] for p in participants]
        drifts = [row[f'{p}_impact_{mode}'] - row[f'{p}_participant_{mode}'] for p in participants]
        
        clip_summaries.append({
            'clip_title': row['clip_title'],
            'Resonance_Rate': np.mean(res_matches),
            'Avg_Drift': np.mean(drifts),
            'Abs_Drift': abs(np.mean(drifts))
        })

    clip_df = pd.DataFrame(clip_summaries)
    
    # C-D: People felt what they saw AND it changed their mood significantly
    cd_top = clip_df[clip_df['Resonance_Rate'] >= 0.6].sort_values(by='Abs_Drift', ascending=False).head(10)
    
    # C-M: People felt what they saw AND it kept their mood stable (Low Drift)
    cm_top = clip_df[clip_df['Resonance_Rate'] >= 0.6].sort_values(by='Abs_Drift', ascending=True).head(10)

    # PRINTING THE REPORT
    print(f"\n{'='*30} {mode.upper()} CONCLUSION REPORT {'='*30}")
    print(pd.DataFrame(user_data).to_string(index=False))
    
    print(f"\n--- TOP 10 C-D CLIPS (Best for INDUCING DRIFT in {mode}) ---")
    print(cd_top[['clip_title', 'Avg_Drift']].to_string(index=False))
    
    print(f"\n--- TOP 10 C-M CLIPS (Best for MAINTAINING STATE in {mode}) ---")
    print(cm_top[['clip_title', 'Avg_Drift']].to_string(index=False))

# Run it
generate_report('valence_btp2.csv', 'valence')
generate_report('arousal_btp2.csv', 'arousal')