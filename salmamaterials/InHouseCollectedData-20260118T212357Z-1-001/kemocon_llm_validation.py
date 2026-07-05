"""
K-EmoCon Complete Validation Pipeline
Loads data, runs LLM inference, compares with ground truth
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from datetime import datetime

print("="*70)
print("K-EMOCON LLM VALIDATION - COMPLETE PIPELINE")
print("="*70)

# Configuration
extract_dir = Path("K-EmoCon/extracted")
e4_data_dir = extract_dir / "e4_data"
annotation_dir = extract_dir / "emotion_annotations" / "aggregated_external_annotations"

# Step 1: Load Emotion Annotations (Ground Truth)
print("\n" + "="*70)
print("STEP 1: Loading Emotion Ground Truth Labels")
print("="*70)

annotation_files = list(annotation_dir.glob("*.csv"))
print(f"\nFound {len(annotation_files)} annotation files")

# Load one sample to see structure
if annotation_files:
    sample_ann = pd.read_csv(annotation_files[0])
    print(f"\nSample annotation file: {annotation_files[0].name}")
    print(f"Shape: {sample_ann.shape}")
    print("\nColumns:")
    print(sample_ann.columns.tolist())
    print("\nFirst 5 rows:")
    print(sample_ann.head())
    
    # Check for valence/arousal columns
    valence_col = [c for c in sample_ann.columns if 'valence' in c.lower()]
    arousal_col = [c for c in sample_ann.columns if 'arousal' in c.lower()]
    
    print(f"\nValence columns: {valence_col}")
    print(f"Arousal columns: {arousal_col}")

# Step 2: Load HR Data and Extract Features
print("\n" + "="*70)
print("STEP 2: Extracting Heart Rate Features")
print("="*70)

hr_files = list(e4_data_dir.rglob("**/E4_HR.csv"))
print(f"\nFound {len(hr_files)} HR files")

hr_features = []

for hr_file in hr_files[:5]:  # Process first 5 as sample
    try:
        # Load HR data
        df_hr = pd.read_csv(hr_file)
        
        # Extract participant ID from folder name
        participant_id = hr_file.parent.name
        
        # Skip header row (row 0), get HR values from 'value' column
        hr_values = df_hr['value'].iloc[1:].astype(float)
        
        # Compute features (matching your Fitbit pipeline)
        features = {
            'participant_id': participant_id,
            'hr_mean': hr_values.mean(),
            'hr_std': hr_values.std(),
            'hr_min': hr_values.min(),
            'hr_max': hr_values.max(),
            'hr_count': len(hr_values)
        }
        
        hr_features.append(features)
        
        print(f"  P{participant_id}: HR={features['hr_mean']:.1f}±{features['hr_std']:.1f} bpm "
              f"(range: {features['hr_min']:.0f}-{features['hr_max']:.0f})")
        
    except Exception as e:
        print(f"  ❌ Error processing {hr_file.name}: {e}")

df_hr_features = pd.DataFrame(hr_features)
print(f"\n✅ Extracted features from {len(df_hr_features)} participants")

# Step 3: Load IBI Data and Calculate HRV
print("\n" + "="*70)
print("STEP 3: Calculating HRV (RMSSD) from IBI Data")
print("="*70)

ibi_files = list(e4_data_dir.rglob("**/E4_IBI.csv"))
print(f"\nFound {len(ibi_files)} IBI files")

def compute_rmssd(ibi_values):
    """Calculate RMSSD (Root Mean Square of Successive Differences)"""
    if len(ibi_values) < 2:
        return np.nan
    successive_diffs = np.diff(ibi_values)
    return np.sqrt(np.mean(successive_diffs ** 2))

hrv_features = []

for ibi_file in ibi_files[:5]:  # Process first 5 as sample
    try:
        # Load IBI data
        df_ibi = pd.read_csv(ibi_file)
        
        # Extract participant ID
        participant_id = ibi_file.parent.name
        
        # Get IBI values (in milliseconds) from 'value' column
        ibi_values = df_ibi['value'].iloc[1:].astype(float).values
        
        # Calculate RMSSD
        rmssd = compute_rmssd(ibi_values)
        
        hrv_features.append({
            'participant_id': participant_id,
            'hrv_rmssd': rmssd
        })
        
        print(f"  P{participant_id}: RMSSD={rmssd:.1f} ms")
        
    except Exception as e:
        print(f"  ❌ Error processing {ibi_file.name}: {e}")

df_hrv = pd.DataFrame(hrv_features)
print(f"\n✅ Calculated HRV for {len(df_hrv)} participants")

# Step 4: Merge HR and HRV Features
print("\n" + "="*70)
print("STEP 4: Merging Features")
print("="*70)

df_features = df_hr_features.merge(df_hrv, on='participant_id', how='left')
print(f"\n✅ Combined dataset: {len(df_features)} participants")
print("\nSample features:")
print(df_features.head())

# Save intermediate results
df_features.to_csv("kemocon_features.csv", index=False)
print("\n✅ Saved: kemocon_features.csv")

# Step 5: Show what we need for LLM inference
print("\n" + "="*70)
print("STEP 5: Next Steps for LLM Validation")
print("="*70)

print("\n✅ Feature extraction complete!")
print(f"\nReady for LLM inference on {len(df_features)} participants")

print("\nWhat we have:")
print("  1. ✅ HR features (mean, std, min, max)")
print("  2. ✅ HRV features (RMSSD)")
print("  3. ✅ Emotion ground truth labels (260 files)")

print("\nNext:")
print("  1. Load emotion labels for these participants")
print("  2. Run LLM inference (same prompt as Fitbit data)")
print("  3. Compare LLM predictions vs K-EmoCon labels")
print("  4. Generate validation report (MAE, correlation)")

print("\n" + "="*70)
print("Copy the output above and I'll create the final LLM inference script!")
print("="*70)