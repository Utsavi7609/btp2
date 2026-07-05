# """
# Generate final_expressed_valence/arousal.csv from BOTH sources
# Creates two sets for comparison
# """

# import pandas as pd
# import numpy as np

# print("="*70)
# print("GENERATING EXPRESSED EMOTIONS - BOTH SOURCES")
# print("="*70)

# # ========================================================================
# # SOURCE 1: VIDEO-EXPRESSED EMOTIONS
# # ========================================================================

# print("\n" + "="*70)
# print("SOURCE 1: VIDEO-EXPRESSED EMOTIONS")
# print("="*70)

# df_video = pd.read_csv("subtitle_video_bin_pred2.csv")

# # Convert 0-1 scale to 1-5 scale
# df_video['clip_id'] = df_video['movie_clip'].str.replace('/', '_')
# df_video['expressed_valence'] = (df_video['val_actual'] * 4) + 1
# df_video['expressed_arousal'] = (df_video['aro_actual'] * 4) + 1

# # Create final files in the format your pipeline expects
# # Assuming your pipeline uses: clip_id as row, users as columns

# # For now, create a simple format with clip_id and emotion values
# df_video_final = df_video[['clip_id', 'expressed_valence', 'expressed_arousal']].copy()

# print(f"\n✅ Processed {len(df_video_final)} video clips")
# print(f"   Valence: {df_video_final['expressed_valence'].mean():.2f} ± {df_video_final['expressed_valence'].std():.2f}")
# print(f"   Arousal: {df_video_final['expressed_arousal'].mean():.2f} ± {df_video_final['expressed_arousal'].std():.2f}")

# # Save VIDEO-expressed versions
# df_video_valence = df_video_final[['clip_id', 'expressed_valence']].copy()
# df_video_arousal = df_video_final[['clip_id', 'expressed_arousal']].copy()

# df_video_valence.to_csv("final_expressed_valence_VIDEO.csv", index=False)
# df_video_arousal.to_csv("final_expressed_arousal_VIDEO.csv", index=False)

# print("\n✅ Saved:")
# print("   - final_expressed_valence_VIDEO.csv")
# print("   - final_expressed_arousal_VIDEO.csv")

# # ========================================================================
# # SOURCE 2: AUDIO-PREDICTED EMOTIONS
# # ========================================================================

# print("\n" + "="*70)
# print("SOURCE 2: AUDIO-PREDICTED EMOTIONS")
# print("="*70)

# df_audio = pd.read_csv("true_music_voice_valence_final.csv")

# # Convert 0-1 scale to 1-5 scale
# df_audio['clip_id'] = df_audio['movie_clip'].str.replace('/', '_')
# df_audio['expressed_valence'] = (df_audio['valence'] * 4) + 1
# df_audio['expressed_arousal'] = (df_audio['arousal'] * 4) + 1

# df_audio_final = df_audio[['clip_id', 'expressed_valence', 'expressed_arousal']].copy()

# print(f"\n✅ Processed {len(df_audio_final)} audio clips")
# print(f"   Valence: {df_audio_final['expressed_valence'].mean():.2f} ± {df_audio_final['expressed_valence'].std():.2f}")
# print(f"   Arousal: {df_audio_final['expressed_arousal'].mean():.2f} ± {df_audio_final['expressed_arousal'].std():.2f}")

# # Save AUDIO-predicted versions
# df_audio_valence = df_audio_final[['clip_id', 'expressed_valence']].copy()
# df_audio_arousal = df_audio_final[['clip_id', 'expressed_arousal']].copy()

# df_audio_valence.to_csv("final_expressed_valence_AUDIO.csv", index=False)
# df_audio_arousal.to_csv("final_expressed_arousal_AUDIO.csv", index=False)

# print("\n✅ Saved:")
# print("   - final_expressed_valence_AUDIO.csv")
# print("   - final_expressed_arousal_AUDIO.csv")

# # ========================================================================
# # COMPARISON
# # ========================================================================

# print("\n" + "="*70)
# print("VIDEO vs AUDIO COMPARISON")
# print("="*70)

# # Merge for comparison
# df_compare = df_video_final.merge(
#     df_audio_final, 
#     on='clip_id', 
#     suffixes=('_video', '_audio')
# )

# print(f"\n✅ Matched {len(df_compare)} clips with both sources")

# # Compute differences
# df_compare['valence_diff'] = abs(df_compare['expressed_valence_video'] - df_compare['expressed_valence_audio'])
# df_compare['arousal_diff'] = abs(df_compare['expressed_arousal_video'] - df_compare['expressed_arousal_audio'])

# print(f"\nMean absolute difference:")
# print(f"  Valence: {df_compare['valence_diff'].mean():.2f}")
# print(f"  Arousal: {df_compare['arousal_diff'].mean():.2f}")

# # Correlation
# from scipy.stats import pearsonr

# corr_v, p_v = pearsonr(df_compare['expressed_valence_video'], df_compare['expressed_valence_audio'])
# corr_a, p_a = pearsonr(df_compare['expressed_arousal_video'], df_compare['expressed_arousal_audio'])

# print(f"\nCorrelation between sources:")
# print(f"  Valence: r={corr_v:.3f} (p={p_v:.4f})")
# print(f"  Arousal: r={corr_a:.3f} (p={p_a:.4f})")

# df_compare.to_csv("video_vs_audio_comparison.csv", index=False)
# print("\n✅ Saved: video_vs_audio_comparison.csv")

# print("\n" + "="*70)
# print("FILES READY FOR Q1-Q5 ANALYSIS")
# print("="*70)
# print("\nNext steps:")
# print("  1. Run Q1-Q5 analysis with VIDEO source")
# print("  2. Run Q1-Q5 analysis with AUDIO source")
# print("  3. Compare results")


"""
Generate final_expressed_valence/arousal.csv from BOTH sources
FIXED version with correct clip ID matching
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("="*70)
print("GENERATING EXPRESSED EMOTIONS - BOTH SOURCES (FIXED)")
print("="*70)

# ========================================================================
# SOURCE 1: VIDEO-EXPRESSED EMOTIONS
# ========================================================================

print("\n" + "="*70)
print("SOURCE 1: VIDEO-EXPRESSED EMOTIONS")
print("="*70)

df_video = pd.read_csv("subtitle_video_bin_pred2.csv")

# FIX: Extract clip ID correctly
# "Gifted/Gifted_035" → "Gifted_035"
def extract_clip_id(movie_clip):
    # Split by / and take the last part
    parts = movie_clip.split('/')
    if len(parts) == 2:
        return parts[1]  # "Gifted_035"
    else:
        return movie_clip  # Already in correct format

df_video['clip_id'] = df_video['movie_clip'].apply(extract_clip_id)

# Convert 0-1 scale to 1-5 scale
df_video['expressed_valence'] = (df_video['val_actual'] * 4) + 1
df_video['expressed_arousal'] = (df_video['aro_actual'] * 4) + 1

df_video_final = df_video[['clip_id', 'expressed_valence', 'expressed_arousal']].copy()

print(f"\n✅ Processed {len(df_video_final)} video clips")
print(f"   Valence: {df_video_final['expressed_valence'].mean():.2f} ± {df_video_final['expressed_valence'].std():.2f}")
print(f"   Arousal: {df_video_final['expressed_arousal'].mean():.2f} ± {df_video_final['expressed_arousal'].std():.2f}")

# Show sample clip IDs
print(f"\nSample clip IDs (first 5):")
for cid in df_video_final['clip_id'].head():
    print(f"  {cid}")

# Save VIDEO-expressed versions
df_video_final.to_csv("final_expressed_valence_VIDEO.csv", index=False)
df_video_final.to_csv("final_expressed_arousal_VIDEO.csv", index=False)

print("\n✅ Saved:")
print("   - final_expressed_valence_VIDEO.csv")
print("   - final_expressed_arousal_VIDEO.csv")

# ========================================================================
# SOURCE 2: AUDIO-PREDICTED EMOTIONS
# ========================================================================

print("\n" + "="*70)
print("SOURCE 2: AUDIO-PREDICTED EMOTIONS")
print("="*70)

df_audio = pd.read_csv("true_music_voice_valence_final.csv")

# Clip ID should already be correct (e.g., "Gifted_035")
df_audio['clip_id'] = df_audio['movie_clip'].str.replace('/', '_')

# Convert 0-1 scale to 1-5 scale
df_audio['expressed_valence'] = (df_audio['valence'] * 4) + 1
df_audio['expressed_arousal'] = (df_audio['arousal'] * 4) + 1

df_audio_final = df_audio[['clip_id', 'expressed_valence', 'expressed_arousal']].copy()

print(f"\n✅ Processed {len(df_audio_final)} audio clips")
print(f"   Valence: {df_audio_final['expressed_valence'].mean():.2f} ± {df_audio_final['expressed_valence'].std():.2f}")
print(f"   Arousal: {df_audio_final['expressed_arousal'].mean():.2f} ± {df_audio_final['expressed_arousal'].std():.2f}")

# Show sample clip IDs
print(f"\nSample clip IDs (first 5):")
for cid in df_audio_final['clip_id'].head():
    print(f"  {cid}")

# Save AUDIO-predicted versions
df_audio_final.to_csv("final_expressed_valence_AUDIO.csv", index=False)
df_audio_final.to_csv("final_expressed_arousal_AUDIO.csv", index=False)

print("\n✅ Saved:")
print("   - final_expressed_valence_AUDIO.csv")
print("   - final_expressed_arousal_AUDIO.csv")

# ========================================================================
# COMPARISON
# ========================================================================

print("\n" + "="*70)
print("VIDEO vs AUDIO COMPARISON")
print("="*70)

# Merge for comparison
df_compare = df_video_final.merge(
    df_audio_final, 
    on='clip_id', 
    suffixes=('_video', '_audio')
)

print(f"\n✅ Matched {len(df_compare)} clips with both sources")

if len(df_compare) > 0:
    # Compute differences
    df_compare['valence_diff'] = abs(df_compare['expressed_valence_video'] - df_compare['expressed_valence_audio'])
    df_compare['arousal_diff'] = abs(df_compare['expressed_arousal_video'] - df_compare['expressed_arousal_audio'])
    
    print(f"\nMean absolute difference:")
    print(f"  Valence: {df_compare['valence_diff'].mean():.2f}")
    print(f"  Arousal: {df_compare['arousal_diff'].mean():.2f}")
    
    # Correlation
    corr_v, p_v = pearsonr(df_compare['expressed_valence_video'], df_compare['expressed_valence_audio'])
    corr_a, p_a = pearsonr(df_compare['expressed_arousal_video'], df_compare['expressed_arousal_audio'])
    
    print(f"\nCorrelation between sources:")
    print(f"  Valence: r={corr_v:.3f} (p={p_v:.4f})")
    print(f"  Arousal: r={corr_a:.3f} (p={p_a:.4f})")
    
    df_compare.to_csv("video_vs_audio_comparison.csv", index=False)
    print("\n✅ Saved: video_vs_audio_comparison.csv")
else:
    print("\n⚠️  No matching clips found!")
    print("\nDebug info:")
    print(f"  Video clips start with: {df_video_final['clip_id'].iloc[0]}")
    print(f"  Audio clips start with: {df_audio_final['clip_id'].iloc[0]}")

print("\n" + "="*70)
print("FILES READY FOR Q1-Q5 ANALYSIS")
print("="*70)