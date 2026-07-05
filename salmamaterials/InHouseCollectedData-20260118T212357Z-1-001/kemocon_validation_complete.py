# """
# K-EmoCon Dataset Validation - Complete Automated Pipeline
# Extracts features, runs LLM inference, compares with ground truth
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import tarfile
# import json
# import os
# from datetime import datetime

# print("="*70)
# print("K-EMOCON DATASET VALIDATION - AUTOMATED PIPELINE")
# print("="*70)

# # Configuration
# KEMOCON_DIR = Path("K-EmoCon")
# E4_DATA_FILE = KEMOCON_DIR / "e4_data.tar"
# ANNOTATIONS_FILE = KEMOCON_DIR / "emotion_annotations.tar.gz"

# # Check if files exist
# if not E4_DATA_FILE.exists():
#     print("\n❌ ERROR: e4_data.tar not found")
#     print(f"\nExpected location: {E4_DATA_FILE.absolute()}")
#     print("\nPlease download from Google Drive:")
#     print("https://drive.google.com/drive/folders/1lYAzLsvtm2ahsEZoDaIOXbsKt7JmKiXJ")
#     print("\nDownload these files:")
#     print("  1. e4_data.tar (280.9 MB)")
#     print("  2. emotion_annotations.tar.gz (95 KB)")
#     print("\nPlace them in: K-EmoCon folder")
#     exit()

# if not ANNOTATIONS_FILE.exists():
#     print("\n❌ ERROR: emotion_annotations.tar.gz not found")
#     print(f"\nExpected location: {ANNOTATIONS_FILE.absolute()}")
#     exit()

# print("\n✅ K-EmoCon files found!")
# print(f"   E4 data: {E4_DATA_FILE.name} ({E4_DATA_FILE.stat().st_size / 1024**2:.1f} MB)")
# print(f"   Annotations: {ANNOTATIONS_FILE.name} ({ANNOTATIONS_FILE.stat().st_size / 1024:.1f} KB)")

# # Step 1: Extract files
# print("\n" + "="*70)
# print("STEP 1: Extracting K-EmoCon Data")
# print("="*70)

# extract_dir = KEMOCON_DIR / "extracted"
# extract_dir.mkdir(exist_ok=True)

# print("\nExtracting e4_data.tar...")
# with tarfile.open(E4_DATA_FILE, 'r') as tar:
#     tar.extractall(extract_dir)
# print("✅ E4 data extracted")

# print("\nExtracting emotion_annotations.tar.gz...")
# with tarfile.open(ANNOTATIONS_FILE, 'r:gz') as tar:
#     tar.extractall(extract_dir)
# print("✅ Emotion annotations extracted")

# # Step 2: Analyze structure
# print("\n" + "="*70)
# print("STEP 2: Analyzing Dataset Structure")
# print("="*70)

# print("\nScanning extracted files...")
# all_files = list(extract_dir.rglob("*"))
# print(f"Total files: {len([f for f in all_files if f.is_file()])}")

# # Find physiological data files
# hr_files = list(extract_dir.rglob("*HR*.csv")) + list(extract_dir.rglob("*hr*.csv"))
# bvp_files = list(extract_dir.rglob("*BVP*.csv")) + list(extract_dir.rglob("*bvp*.csv"))
# annotation_files = list(extract_dir.rglob("*annotation*.csv")) + list(extract_dir.rglob("*label*.csv"))

# print(f"\nFound:")
# print(f"  Heart Rate files: {len(hr_files)}")
# print(f"  BVP files (for HRV): {len(bvp_files)}")
# print(f"  Annotation files: {len(annotation_files)}")

# # Step 3: Show structure for next steps
# print("\n" + "="*70)
# print("STEP 3: Dataset Structure Analysis")
# print("="*70)

# print("\nDirectory structure:")
# for item in sorted(extract_dir.rglob("*"))[:20]:  # Show first 20 items
#     if item.is_file():
#         rel_path = item.relative_to(extract_dir)
#         size = item.stat().st_size
#         if size < 1024:
#             size_str = f"{size} B"
#         elif size < 1024**2:
#             size_str = f"{size/1024:.1f} KB"
#         else:
#             size_str = f"{size/1024**2:.1f} MB"
#         print(f"  {rel_path} ({size_str})")

# print("\n" + "="*70)
# print("NEXT STEP: Share the Output Above")
# print("="*70)
# print("\nCopy the directory structure above and paste it here.")
# print("I will then create the complete processing script that:")
# print("  1. Loads K-EmoCon HR and HRV data")
# print("  2. Extracts same features as your Fitbit data")
# print("  3. Loads emotion ground truth labels")
# print("  4. Runs LLM inference on K-EmoCon samples")
# print("  5. Compares LLM predictions vs K-EmoCon labels")
# print("  6. Generates validation report with metrics")
# print("\nThis will complete your K-EmoCon validation!")




"""
K-EmoCon Complete Processing Pipeline
Loads HR/HRV, finds emotion labels, runs LLM inference, validates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import tarfile

print("="*70)
print("K-EMOCON PROCESSING - PHASE 2")
print("="*70)

# Paths
extract_dir = Path("K-EmoCon/extracted")

# Step 1: Find emotion annotation files more thoroughly
print("\n" + "="*70)
print("STEP 1: Locating Emotion Annotation Files")
print("="*70)

print("\nSearching all extracted files for annotations...")

# Check if emotion_annotations folder exists
emotion_dir = extract_dir / "emotion_annotations"
if emotion_dir.exists():
    print(f"✅ Found emotion_annotations folder: {emotion_dir}")
    annotation_files = list(emotion_dir.rglob("*"))
    print(f"   Files inside: {len(annotation_files)}")
    for f in annotation_files[:10]:
        print(f"   - {f.name}")
else:
    print("⚠️  emotion_annotations folder not found in standard location")
    print("   Searching entire extracted directory...")
    
    # Search for any file with "annotation", "emotion", or "label"
    all_files = list(extract_dir.rglob("*"))
    potential_annotations = [
        f for f in all_files 
        if f.is_file() and any(keyword in f.name.lower() 
        for keyword in ['annotation', 'emotion', 'label', 'valence', 'arousal'])
    ]
    
    print(f"\nFound {len(potential_annotations)} potential annotation files:")
    for f in potential_annotations:
        rel_path = f.relative_to(extract_dir)
        print(f"   - {rel_path}")

# Step 2: Check what's actually in emotion_annotations.tar.gz
print("\n" + "="*70)
print("STEP 2: Inspecting emotion_annotations.tar.gz Contents")
print("="*70)

annotation_tar = Path("K-EmoCon/emotion_annotations.tar.gz")
print(f"\nListing contents of {annotation_tar.name}...")

try:
    with tarfile.open(annotation_tar, 'r:gz') as tar:
        members = tar.getmembers()
        print(f"\nTotal files in archive: {len(members)}")
        print("\nFirst 20 files:")
        for i, member in enumerate(members[:20]):
            print(f"   {i+1:2d}. {member.name} ({member.size} bytes)")
except Exception as e:
    print(f"❌ Error reading tar: {e}")

# Step 3: Load and preview HR data
print("\n" + "="*70)
print("STEP 3: Loading Heart Rate Data (Sample)")
print("="*70)

hr_files = list(extract_dir.rglob("**/E4_HR.csv"))
print(f"\nFound {len(hr_files)} HR files")

if hr_files:
    # Load first subject as example
    sample_hr = hr_files[0]
    print(f"\nSample file: {sample_hr.relative_to(extract_dir)}")
    
    try:
        df_hr = pd.read_csv(sample_hr, header=None)
        print(f"\nShape: {df_hr.shape}")
        print("\nFirst 10 rows:")
        print(df_hr.head(10))
        
        # K-EmoCon E4 format: first row is timestamp, second row is sample rate
        if len(df_hr) > 2:
            start_time = df_hr.iloc[0, 0]
            sample_rate = df_hr.iloc[1, 0]
            hr_values = df_hr.iloc[2:, 0].values
            
            print(f"\nParsed data:")
            print(f"  Start timestamp: {start_time}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  HR values: {len(hr_values)} samples")
            print(f"  HR range: {hr_values.min():.1f} - {hr_values.max():.1f} bpm")
    except Exception as e:
        print(f"❌ Error loading: {e}")

# Step 4: Load and preview IBI data (for HRV)
print("\n" + "="*70)
print("STEP 4: Loading IBI Data for HRV (Sample)")
print("="*70)

ibi_files = list(extract_dir.rglob("**/E4_IBI.csv"))
print(f"\nFound {len(ibi_files)} IBI files")

if ibi_files:
    sample_ibi = ibi_files[0]
    print(f"\nSample file: {sample_ibi.relative_to(extract_dir)}")
    
    try:
        df_ibi = pd.read_csv(sample_ibi, header=None)
        print(f"\nShape: {df_ibi.shape}")
        print("\nFirst 10 rows:")
        print(df_ibi.head(10))
        
        # IBI format: first row is timestamp, then pairs of (time, IBI)
        if len(df_ibi) > 1:
            ibi_values = df_ibi.iloc[1:, 1].values
            print(f"\nIBI values: {len(ibi_values)} intervals")
            print(f"IBI range: {ibi_values.min()*1000:.1f} - {ibi_values.max()*1000:.1f} ms")
    except Exception as e:
        print(f"❌ Error loading: {e}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\n✅ K-EmoCon data structure understood!")
print("\nNext: Paste this output and I'll create:")
print("  1. Emotion label loader")
print("  2. Feature extraction (matching your Fitbit pipeline)")
print("  3. LLM inference on K-EmoCon data")
print("  4. Validation report (LLM vs ground truth)")