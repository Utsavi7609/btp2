import pandas as pd
import random
import os

df = pd.read_csv("d:/BTP/btp2/salmamaterials/final_expressed_valence_AUDIO.csv")
df_a = pd.read_csv("d:/BTP/btp2/salmamaterials/final_expressed_arousal_AUDIO.csv")
df = df.merge(df_a, on='clip_id')

def get_quad(v, a):
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"

df['quad'] = df.apply(lambda row: get_quad(row['expressed_valence'], row['expressed_arousal']), axis=1)

q1_clips = df[df['quad'] == 'Q1']['clip_id'].tolist()
q2_clips = df[df['quad'] == 'Q2']['clip_id'].tolist()
q3_clips = df[df['quad'] == 'Q3']['clip_id'].tolist()
q4_clips = df[df['quad'] == 'Q4']['clip_id'].tolist()
q0_clips = df[df['quad'] == 'Q0']['clip_id'].tolist()

def generate_playlist(user_id):
    blocks = [
        ('Q4_Maintenance_Block', [random.choice(q4_clips), random.choice(q4_clips)]),
        ('Drift_to_Q4_Block', [random.choice(q2_clips + q3_clips), random.choice(q4_clips)]),
        ('Hyp9_Transition_Test', [random.choice(q1_clips + q2_clips + q3_clips), random.choice(q0_clips)]),
        ('Hyp9_Maintenance_Test', [random.choice(q0_clips), random.choice(q0_clips)])
    ]
    random.shuffle(blocks)
    
    playlist = []
    order = 1
    for b_type, clips in blocks:
        playlist.append({'user_id': user_id, 'clip_order': order, 'clip_title': f"{clips[0]}.mp4", 'block_type': b_type, 'expected_role': 'Clip1_Stimulus'})
        order += 1
        playlist.append({'user_id': user_id, 'clip_order': order, 'clip_title': f"{clips[1]}.mp4", 'block_type': b_type, 'expected_role': 'Clip2_Target'})
        order += 1
        
    return playlist

playlists = []
for uid in range(1, 16):
    playlists.extend(generate_playlist(uid))

out_df = pd.DataFrame(playlists)
out_df.to_csv("d:/BTP/btp2/salmamaterials/_validation_playlists_assigned.csv", index=False)
print("Generated _validation_playlists_assigned.csv for 15 users.")
