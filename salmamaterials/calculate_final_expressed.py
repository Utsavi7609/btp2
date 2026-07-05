# import pandas as pd

# # 1. Load your existing data files
# useful_df = pd.read_csv('clip_usefulness_labels.csv')
# expressed_df = pd.read_csv('true_music_voice_valence_final.csv')
# p_val_df = pd.read_csv('perceived_valence_summary.csv')
# p_aro_df = pd.read_csv('perceived_arousal_summary.csv')

# # 2. Prepare the data (Normalize IDs and Scale)
# expressed_df['clip_id'] = expressed_df['movie_clip'].str.replace('.mp4', '', case=False).str.strip()
# useful_df['clip_id'] = useful_df['clip_id'].str.replace('.mp4', '', case=False).str.strip()

# # Rescale expressed emotions to 1-5 scale: 1 + (X * 4)
# # Video part
# expressed_df['V_vdo'] = 1 + expressed_df['valence'] * 4
# expressed_df['A_vdo'] = 1 + expressed_df['arousal'] * 4
# # Audio part
# expressed_df['V_aud'] = 1 + expressed_df['val_actual'] * 4
# expressed_df['A_aud'] = 1 + expressed_df['aro_actual'] * 4

# # Merge useful labels and perceived averages into a master processing table
# master = pd.merge(useful_df, expressed_df[['clip_id', 'V_vdo', 'A_vdo', 'V_aud', 'A_aud']], on='clip_id')
# master = pd.merge(master, p_val_df[['clip_id', 'avg_valence']], on='clip_id')
# master = pd.merge(master, p_aro_df[['clip_id', 'avg_arousal']], on='clip_id')

# ignored_clips = []
# final_v = []
# final_a = []

# # 3. Apply the 3 Cases
# for _, row in master.iterrows():
#     cid = row['clip_id']
#     au = row['audio_useful']
#     vu = row['video_useful']
    
#     # CASE 1: Both not useful -> Ignore
#     if au == 0 and vu == 0:
#         ignored_clips.append(cid)
#         continue

#     v_final, a_final = None, None

#     # CASE 2: Only one is useful
#     if au == 1 and vu == 0:
#         v_final, a_final = row['V_aud'], row['A_aud']
#     elif au == 0 and vu == 1:
#         v_final, a_final = row['V_vdo'], row['A_vdo']

#     # CASE 3: Both are useful -> Match against Perceived
#     elif au == 1 and vu == 1:
#         v_per = row['avg_valence']
#         a_per = row['avg_arousal']
        
#         # Match condition: |perceived - expressed| <= 1 for BOTH V and A
#         audio_match = (abs(v_per - row['V_aud']) <= 1) and (abs(a_per - row['A_aud']) <= 1)
#         video_match = (abs(v_per - row['V_vdo']) <= 1) and (abs(a_per - row['A_vdo']) <= 1)
        
#         if audio_match and not video_match:
#             v_final, a_final = row['V_aud'], row['A_aud']
#         elif video_match and not audio_match:
#             v_final, a_final = row['V_vdo'], row['A_vdo']
#         else:
#             # If both match or neither match, pick the one with the smallest total error
#             aud_err = abs(v_per - row['V_aud']) + abs(a_per - row['A_aud'])
#             vdo_err = abs(v_per - row['V_vdo']) + abs(a_per - row['A_vdo'])
#             if aud_err <= vdo_err:
#                 v_final, a_final = row['V_aud'], row['A_aud']
#             else:
#                 v_final, a_final = row['V_vdo'], row['A_vdo']

#     if v_final is not None:
#         final_v.append({'clip_id': cid, 'expressed_value': round(v_final, 3)})
#         final_a.append({'clip_id': cid, 'expressed_value': round(a_final, 3)})

# # 4. Save Outputs
# pd.DataFrame({'clip_id': ignored_clips}).to_csv('ignored_clips_list.csv', index=False)
# pd.DataFrame(final_v).to_csv('final_expressed_valence.csv', index=False)
# pd.DataFrame(final_a).to_csv('final_expressed_arousal.csv', index=False)

# print(f"Process complete. Ignored {len(ignored_clips)} clips.")

import pandas as pd
import os

def clean_id(series):
    """Removes .mp4 and extra spaces to ensure matching."""
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# 1. Load your existing data files
print("Loading files...")
useful_df = pd.read_csv('clip_usefulness_labels.csv')
expressed_df = pd.read_csv('true_music_voice_valence_final.csv')

# Check if the perceived summary files exist
p_val_file = 'perceived_valence_summary.csv'
p_aro_file = 'perceived_arousal_summary.csv'

if not os.path.exists(p_val_file) or not os.path.exists(p_aro_file):
    print(f"ERROR: Missing {p_val_file} or {p_aro_file}.")
    print("Please generate these using the previous summary script first!")
else:
    p_val_df = pd.read_csv(p_val_file)
    p_aro_df = pd.read_csv(p_aro_file)

    # 2. CLEAN ALL IDs (The Fix)
    useful_df['clip_id'] = clean_id(useful_df['clip_id'])
    expressed_df['clip_id'] = clean_id(expressed_df['movie_clip'])
    p_val_df['clip_id'] = clean_id(p_val_df['clip_id'])
    p_aro_df['clip_id'] = clean_id(p_aro_df['clip_id'])

    # 3. Rescale expressed emotions to 1-5 scale: 1 + (X * 4)
    expressed_df['V_vdo'] = 1 + expressed_df['valence'] * 4
    expressed_df['A_vdo'] = 1 + expressed_df['arousal'] * 4
    expressed_df['V_aud'] = 1 + expressed_df['val_actual'] * 4
    expressed_df['A_aud'] = 1 + expressed_df['aro_actual'] * 4

    # 4. Merge data safely
    # We use 'inner' join so we only process clips that exist in ALL files
    master = pd.merge(useful_df, expressed_df[['clip_id', 'V_vdo', 'A_vdo', 'V_aud', 'A_aud']], on='clip_id')
    master = pd.merge(master, p_val_df[['clip_id', 'avg_valence']], on='clip_id')
    master = pd.merge(master, p_aro_df[['clip_id', 'avg_arousal']], on='clip_id')

    print(f"Total clips found for processing: {len(master)}")

    ignored_clips = []
    final_v = []
    final_a = []

    # 5. Apply the 3 Cases
    for _, row in master.iterrows():
        cid = row['clip_id']
        au = row['audio_useful']
        vu = row['video_useful']
        
        # CASE 1: Both not useful -> Ignore
        if au == 0 and vu == 0:
            ignored_clips.append(cid)
            continue

        v_final, a_final = None, None

        # CASE 2: Only one is useful
        if au == 1 and vu == 0:
            v_final, a_final = row['V_aud'], row['A_aud']
        elif au == 0 and vu == 1:
            v_final, a_final = row['V_vdo'], row['A_vdo']

        # CASE 3: Both are useful -> Match against Perceived
        elif au == 1 and vu == 1:
            v_per = row['avg_valence']
            a_per = row['avg_arousal']
            
            # Match condition: |perceived - expressed| <= 1 for BOTH
            audio_match = (abs(v_per - row['V_aud']) <= 1) and (abs(a_per - row['A_aud']) <= 1)
            video_match = (abs(v_per - row['V_vdo']) <= 1) and (abs(a_per - row['A_vdo']) <= 1)
            
            if audio_match and not video_match:
                v_final, a_final = row['V_aud'], row['A_aud']
            elif video_match and not audio_match:
                v_final, a_final = row['V_vdo'], row['A_vdo']
            else:
                # Tie-breaker or neither match: Pick the one closer to human perception
                aud_err = abs(v_per - row['V_aud']) + abs(a_per - row['A_aud'])
                vdo_err = abs(v_per - row['V_vdo']) + abs(a_per - row['A_vdo'])
                if aud_err <= vdo_err:
                    v_final, a_final = row['V_aud'], row['A_aud']
                else:
                    v_final, a_final = row['V_vdo'], row['A_vdo']

        if v_final is not None:
            final_v.append({'clip_id': cid, 'expressed_value': round(v_final, 3)})
            final_a.append({'clip_id': cid, 'expressed_value': round(a_final, 3)})

    # 6. Save Outputs
    pd.DataFrame({'clip_id': ignored_clips}).to_csv('ignored_clips_list.csv', index=False)
    pd.DataFrame(final_v).to_csv('final_expressed_valence.csv', index=False)
    pd.DataFrame(final_a).to_csv('final_expressed_arousal.csv', index=False)

    print(f"Success! Ignored: {len(ignored_clips)} | Processed: {len(final_v)}")
    print("Files 'final_expressed_valence.csv' and 'final_expressed_arousal.csv' created.")