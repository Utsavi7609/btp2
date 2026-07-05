"""
Rebuild inferred_valence_summary.csv and inferred_arousal_summary.csv
by matching Fitbit physiological data to clip viewing timestamps
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("REBUILDING CLIP-LEVEL INFERRED EMOTIONS")
print("="*70)

# ========================================================================
# STEP 1: Load Clip Viewing Log
# ========================================================================

print("\n1. Loading clip viewing log...")

clip_log = pd.read_excel("D:/BTP/btp2/files/input_files/clip_responses_export.xlsx")

# Convert timestamp
clip_log['created_at'] = pd.to_datetime(clip_log['created_at'])

# Clean clip title (remove .mp4)
clip_log['clip_id'] = clip_log['clip_title'].str.replace('.mp4', '', regex=False)

print(f"✅ Loaded {len(clip_log)} clip viewing records")
print(f"   Participants: {clip_log['participant'].nunique()}")
print(f"   Unique clips: {clip_log['clip_id'].nunique()}")

# Show sample
print("\nSample viewing log:")
print(clip_log[['participant', 'clip_id', 'created_at']].head(10))

# ========================================================================
# STEP 2: Load Fitbit Physiological Windows
# ========================================================================

print("\n2. Loading Fitbit physiological windows...")

# Adjust this path to where your Fitbit HR windows are
fitbit_path = "D:/BTP/btp2/salmamaterials/InHouseCollectedData-20260118T212357Z-1-001/"

try:
    df_fitbit = pd.read_csv(fitbit_path + "fitbit_hr_windows.csv")
    print(f"✅ Loaded {len(df_fitbit)} physiological windows")
except:
    print("❌ Could not find fitbit_hr_windows.csv")
    print(f"   Expected location: {fitbit_path}")
    print("\nPlease update the fitbit_path variable in the script")
    exit()

# Convert date to datetime
df_fitbit['date'] = pd.to_datetime(df_fitbit['date'])

print("\nSample Fitbit data:")
print(df_fitbit.head())

# ========================================================================
# STEP 3: Load LLM Inferred Emotions (Per Window)
# ========================================================================

print("\n3. Loading LLM inferred emotions...")

try:
    df_llm = pd.read_csv(fitbit_path + "inferred_emotions_llm.csv")
    print(f"✅ Loaded {len(df_llm)} LLM emotion inferences")
except:
    print("❌ Could not find inferred_emotions_llm.csv")
    exit()

# Convert date to datetime
df_llm['date'] = pd.to_datetime(df_llm['date'])

# Merge Fitbit + LLM
df_phys = df_fitbit.merge(df_llm[['user_id', 'date', 'inferred_valence', 'inferred_arousal']], 
                          on=['user_id', 'date'], 
                          how='left')

print(f"✅ Merged physiological + LLM data: {len(df_phys)} windows")

# ========================================================================
# STEP 4: Match Physiological Windows to Clip Timestamps
# ========================================================================

print("\n4. Matching physiological data to clip viewing timestamps...")

def find_matching_window(user, clip_time, df_phys, time_window_minutes=120):
    """
    Find physiological window that overlaps with clip viewing time
    """
    user_data = df_phys[df_phys['user_id'] == user].copy()
    
    if len(user_data) == 0:
        return None
    
    # Calculate time differences
    user_data['time_diff'] = abs((user_data['date'] - clip_time).dt.total_seconds() / 60)
    
    # Find closest window within time_window_minutes
    closest = user_data[user_data['time_diff'] <= time_window_minutes].sort_values('time_diff')
    
    if len(closest) > 0:
        return closest.iloc[0]
    else:
        return None

# Match each clip viewing to physiological data
matched_data = []

for idx, row in clip_log.iterrows():
    user = row['participant']
    clip_id = row['clip_id']
    clip_time = row['created_at']
    
    # Find matching physiological window
    phys_window = find_matching_window(user, clip_time, df_phys)
    
    if phys_window is not None:
        matched_data.append({
            'clip_id': clip_id,
            'user_id': user,
            'clip_time': clip_time,
            'phys_time': phys_window['date'],
            'inferred_valence': phys_window['inferred_valence'],
            'inferred_arousal': phys_window['inferred_arousal'],
            'hr_mean': phys_window['hr_mean'],
            'hrv_rmssd': phys_window['hrv_rmssd']
        })

df_matched = pd.DataFrame(matched_data)

print(f"✅ Matched {len(df_matched)} clips to physiological data")
print(f"   Match rate: {len(df_matched)/len(clip_log)*100:.1f}%")

# ========================================================================
# STEP 5: Aggregate by Clip
# ========================================================================

print("\n5. Aggregating emotions by clip...")

# Create pivot tables
valence_summary = df_matched.pivot_table(
    index='clip_id',
    columns='user_id',
    values='inferred_valence',
    aggfunc='mean'
)

arousal_summary = df_matched.pivot_table(
    index='clip_id',
    columns='user_id',
    values='inferred_arousal',
    aggfunc='mean'
)

# Calculate averages
valence_summary['avg_inferred_valence'] = valence_summary.mean(axis=1)
arousal_summary['avg_inferred_arousal'] = arousal_summary.mean(axis=1)

# Reset index to make clip_id a column
valence_summary = valence_summary.reset_index()
arousal_summary = arousal_summary.reset_index()

print(f"✅ Created summaries:")
print(f"   Valence: {len(valence_summary)} clips")
print(f"   Arousal: {len(arousal_summary)} clips")

# ========================================================================
# STEP 6: Save NEW Inferred Summaries
# ========================================================================

print("\n6. Saving NEW inferred emotion summaries...")

# Save to salmamaterials directory (where your analysis scripts expect them)
output_path = "D:/BTP/btp2/salmamaterials/"

valence_summary.to_csv(output_path + "inferred_valence_summary_NEW.csv", index=False)
arousal_summary.to_csv(output_path + "inferred_arousal_summary_NEW.csv", index=False)

print(f"✅ Saved:")
print(f"   - {output_path}inferred_valence_summary_NEW.csv")
print(f"   - {output_path}inferred_arousal_summary_NEW.csv")

# Show sample
print("\n" + "="*70)
print("SAMPLE VALENCE SUMMARY (First 10 clips)")
print("="*70)
print(valence_summary.head(10))

print("\n" + "="*70)
print("REBUILD COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Verify the NEW files look correct")
print("2. Copy them to replace the old files:")
print(f"   copy inferred_valence_summary_NEW.csv inferred_valence_summary.csv")
print(f"   copy inferred_arousal_summary_NEW.csv inferred_arousal_summary.csv")
print("3. Run the threshold analysis script")