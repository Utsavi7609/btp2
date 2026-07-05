"""Task H: Gather validation video files into a single export folder."""
import json
import os
import shutil

BASE = r"d:\BTP\btp2\salmamaterials"
SEARCH_DIRS = [
    os.path.join(BASE, "fitbit_setup", "emotionlab", "media", "clips"),
    r"D:\BTP\btp1\mg_vdo",
]
EXPORT_DIR = os.path.join(BASE, "_validation_videos_export")
PLAYLISTS_JSON = os.path.join(BASE, "_validation_playlists.json")

def main():
    # 1. Create/clear export folder
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR)
    print(f"Export folder ready: {EXPORT_DIR}")

    # 2. Read target clips from playlists JSON
    with open(PLAYLISTS_JSON, "r", encoding="utf-8") as f:
        playlists = json.load(f)

    target_clips = set()
    for playlist in playlists:
        for entry in playlist["sequence"]:
            target_clips.add(entry["clip_title"])

    print(f"Unique clips required: {len(target_clips)}")

    # 3. Recursively search all clip directories and copy matches
    # Build index of all mp4 files across all search paths
    found_map = {}  # clip_title -> full_path
    for search_dir in SEARCH_DIRS:
        if not os.path.exists(search_dir):
            print(f"  WARNING: {search_dir} does not exist, skipping.")
            continue
        for root, dirs, files in os.walk(search_dir):
            for fname in files:
                if fname in target_clips and fname not in found_map:
                    found_map[fname] = os.path.join(root, fname)

    # 4. Copy found files
    copied = 0
    for clip_title, src_path in sorted(found_map.items()):
        dst_path = os.path.join(EXPORT_DIR, clip_title)
        shutil.copy2(src_path, dst_path)
        copied += 1
        print(f"  Copied: {clip_title}")

    # 5. Report
    missing = target_clips - set(found_map.keys())
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Required: {len(target_clips)}")
    print(f"  Found & copied: {copied}")
    print(f"  Missing: {len(missing)}")
    if missing:
        print(f"\n  MISSING CLIPS:")
        for m in sorted(missing):
            print(f"    - {m}")
    else:
        print(f"\n  All clips found successfully!")

if __name__ == "__main__":
    main()
