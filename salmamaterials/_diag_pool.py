"""Diagnose why pool selection only picked 2 movies."""
import pandas as pd
import numpy as np

BASE = r"d:\BTP\btp2\salmamaterials"
df = pd.read_csv(f"{BASE}\\_consecutive_pair_dataset_AUDIO.csv")

all_clips = set(df["clip1_title"].unique()) | set(df["clip2_title"].unique())
print(f"Total unique clips: {len(all_clips)}")

movies = {}
for c in all_clips:
    movie = c.rsplit("_", 1)[0]
    movies.setdefault(movie, []).append(c)

print(f"\nMovies ({len(movies)}):")
for m in sorted(movies.keys()):
    print(f"  {m}: {len(movies[m])} clips")

# Build per-clip quadrant reliability
clip_records = []
for _, r in df.iterrows():
    clip_records.append({"clip_title": r["clip1_title"], "E_quad": r["clip1_E_quad"]})
    clip_records.append({"clip_title": r["clip2_title"], "E_quad": r["clip2_E_quad"]})
clips_df = pd.DataFrame(clip_records)

quad_counts = clips_df.groupby(["clip_title", "E_quad"]).size().unstack(fill_value=0)
for q in ["Q1", "Q2", "Q3", "Q4", "Q0"]:
    if q not in quad_counts.columns:
        quad_counts[q] = 0

quad_counts["dominant_quad"] = quad_counts[["Q1", "Q2", "Q3", "Q4", "Q0"]].idxmax(axis=1)
quad_counts["dominant_count"] = quad_counts[["Q1", "Q2", "Q3", "Q4", "Q0"]].max(axis=1)
quad_counts["total"] = quad_counts[["Q1", "Q2", "Q3", "Q4", "Q0"]].sum(axis=1)
quad_counts["reliability"] = quad_counts["dominant_count"] / quad_counts["total"]

print(f"\nReliability stats:")
print(f"  All 1.0: {(quad_counts['reliability'] == 1.0).sum()}")
print(f"  < 1.0:   {(quad_counts['reliability'] < 1.0).sum()}")
print(f"\nAppearance counts:")
print(quad_counts["total"].value_counts().sort_index().to_string())

# THE BUG: if all clips have reliability=1.0 and appearances=12,
# sort_values is stable => alphabetical order wins => AboutTime first!
print("\n\n=== THE ROOT CAUSE ===")
print("All clips have identical reliability (1.0) and appearances (12).")
print("sort_values is stable, so alphabetical order determines selection.")
print("'AboutTime' comes first alphabetically => all pools filled from AboutTime.")
print("TheBlindSide only appeared in Neutral pool because its E_val/E_aro are near 3.0.")

# Show what a DIVERSE selection would look like
print("\n=== DIVERSE SELECTION (by movie) ===")
for target_q in ["Q4", "Q3", "Q2", "Q1"]:
    q_clips = quad_counts[quad_counts["dominant_quad"] == target_q].copy()
    q_clips["movie"] = [c.rsplit("_", 1)[0] for c in q_clips.index]
    print(f"\n  {target_q} clips by movie:")
    print(q_clips.groupby("movie").size().sort_values(ascending=False).to_string())
