# """
# Generate summary CSVs for easy analysis
# """

# import pandas as pd

# df = pd.read_csv("inferred_emotions_llm.csv")

# # Valence summary
# valence_summary = df.pivot_table(
#     index="date",
#     columns="user_id",
#     values="inferred_valence",
#     aggfunc="mean"
# )
# valence_summary["average"] = valence_summary.mean(axis=1)
# valence_summary.to_csv("inferred_valence_summary.csv")

# # Arousal summary
# arousal_summary = df.pivot_table(
#     index="date",
#     columns="user_id",
#     values="inferred_arousal",
#     aggfunc="mean"
# )
# arousal_summary["average"] = arousal_summary.mean(axis=1)
# arousal_summary.to_csv("inferred_arousal_summary.csv")

# print("✅ Created:")
# print("  - inferred_valence_summary.csv")
# print("  - inferred_arousal_summary.csv")


"""
Generate Summary CSVs and Statistics
Creates the files mentor expects
"""

import pandas as pd
import numpy as np

df = pd.read_csv("inferred_emotions_llm.csv")

print("="*70)
print("GENERATING SUMMARY FILES")
print("="*70)

# 1. Valence Summary
print("\nCreating valence summary...")
valence_summary = df.pivot_table(
    index="date",
    columns="user_id",
    values="inferred_valence",
    aggfunc="mean"
)
valence_summary["avg_inferred_valence"] = valence_summary.mean(axis=1)
valence_summary.to_csv("inferred_valence_summary.csv")
print("✅ inferred_valence_summary.csv")

# 2. Arousal Summary
print("Creating arousal summary...")
arousal_summary = df.pivot_table(
    index="date",
    columns="user_id",
    values="inferred_arousal",
    aggfunc="mean"
)
arousal_summary["avg_inferred_arousal"] = arousal_summary.mean(axis=1)
arousal_summary.to_csv("inferred_arousal_summary.csv")
print("✅ inferred_arousal_summary.csv")

# 3. Comprehensive Statistics
print("\nCreating comprehensive statistics...")

stats = {
    'Metric': [],
    'Valence': [],
    'Arousal': []
}

for metric in ['mean', 'std', 'min', 'max', 'median']:
    stats['Metric'].append(metric)
    stats['Valence'].append(getattr(df['inferred_valence'], metric)())
    stats['Arousal'].append(getattr(df['inferred_arousal'], metric)())

stats_df = pd.DataFrame(stats)
stats_df.to_csv("emotion_statistics.csv", index=False)
print("✅ emotion_statistics.csv")

# 4. Per-user summary
print("Creating per-user summary...")
user_summary = df.groupby('user_id').agg({
    'inferred_valence': ['mean', 'std', 'min', 'max'],
    'inferred_arousal': ['mean', 'std', 'min', 'max'],
    'hr_mean': 'mean',
    'hrv_rmssd': 'mean'
}).round(2)
user_summary.to_csv("user_emotion_summary.csv")
print("✅ user_emotion_summary.csv")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTotal samples: {len(df)}")
print(f"Users: {df['user_id'].nunique()}")
print(f"\nValence: {df['inferred_valence'].mean():.2f} ± {df['inferred_valence'].std():.2f}")
print(f"Arousal: {df['inferred_arousal'].mean():.2f} ± {df['inferred_arousal'].std():.2f}")

print("\n✅ All summaries generated!")