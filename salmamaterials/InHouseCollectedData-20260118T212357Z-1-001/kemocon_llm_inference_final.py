# """
# K-EmoCon LLM Inference and Validation
# Runs LLM on all participants, compares with ground truth
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import os
# from datetime import datetime
# import requests
# import json
# import time

# print("="*70)
# print("K-EMOCON LLM INFERENCE - FINAL VALIDATION")
# print("="*70)

# # Configuration
# extract_dir = Path("K-EmoCon/extracted")
# e4_data_dir = extract_dir / "e4_data"
# annotation_dir = extract_dir / "emotion_annotations" / "aggregated_external_annotations"

# GROQ_API_KEY = "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"
# MODEL = "llama-3.3-70b-versatile"

# # Step 1: Extract Features from ALL Participants
# print("\n" + "="*70)
# print("STEP 1: Extracting Features from All Participants")
# print("="*70)

# def compute_rmssd(ibi_values):
#     """Calculate RMSSD"""
#     if len(ibi_values) < 2:
#         return np.nan
#     successive_diffs = np.diff(ibi_values)
#     return np.sqrt(np.mean(successive_diffs ** 2))

# # Process all HR files
# hr_files = list(e4_data_dir.rglob("**/E4_HR.csv"))
# print(f"\nProcessing {len(hr_files)} participants...")

# hr_features = []

# for hr_file in hr_files:
#     try:
#         df_hr = pd.read_csv(hr_file)
#         participant_id = hr_file.parent.name
#         hr_values = df_hr['value'].iloc[1:].astype(float)
        
#         features = {
#             'participant_id': participant_id,
#             'hr_mean': hr_values.mean(),
#             'hr_std': hr_values.std(),
#             'hr_min': hr_values.min(),
#             'hr_max': hr_values.max(),
#             'hr_count': len(hr_values)
#         }
        
#         hr_features.append(features)
        
#     except Exception as e:
#         print(f"  ⚠️  Error P{participant_id}: {e}")

# df_hr_features = pd.DataFrame(hr_features)
# print(f"✅ Extracted HR features: {len(df_hr_features)} participants")

# # Process all IBI files
# ibi_files = list(e4_data_dir.rglob("**/E4_IBI.csv"))

# hrv_features = []

# for ibi_file in ibi_files:
#     try:
#         df_ibi = pd.read_csv(ibi_file)
#         participant_id = ibi_file.parent.name
#         ibi_values = df_ibi['value'].iloc[1:].astype(float).values
#         rmssd = compute_rmssd(ibi_values)
        
#         hrv_features.append({
#             'participant_id': participant_id,
#             'hrv_rmssd': rmssd
#         })
        
#     except Exception as e:
#         pass

# df_hrv = pd.DataFrame(hrv_features)
# print(f"✅ Calculated HRV: {len(df_hrv)} participants")

# # Merge features
# df_features = df_hr_features.merge(df_hrv, on='participant_id', how='left')
# df_features['hrv_rmssd'] = df_features['hrv_rmssd'].fillna(df_features['hrv_rmssd'].mean())

# print(f"\n✅ Total features ready: {len(df_features)} participants")
# print(f"\nFeature summary:")
# print(f"  HR mean: {df_features['hr_mean'].mean():.1f} ± {df_features['hr_mean'].std():.1f} bpm")
# print(f"  RMSSD: {df_features['hrv_rmssd'].mean():.1f} ± {df_features['hrv_rmssd'].std():.1f} ms")

# # Step 2: Load Ground Truth Emotion Labels
# print("\n" + "="*70)
# print("STEP 2: Loading Ground Truth Emotion Labels")
# print("="*70)

# annotation_files = list(annotation_dir.glob("*.csv"))

# ground_truth = []

# for ann_file in annotation_files:
#     try:
#         participant_id = ann_file.stem.replace('P', '').replace('.external', '')
#         df_ann = pd.read_csv(ann_file)
        
#         # Get average valence and arousal across all timepoints
#         avg_valence = df_ann['valence'].mean()
#         avg_arousal = df_ann['arousal'].mean()
        
#         ground_truth.append({
#             'participant_id': participant_id,
#             'true_valence': avg_valence,
#             'true_arousal': avg_arousal
#         })
        
#     except Exception as e:
#         print(f"  ⚠️  Error loading {ann_file.name}: {e}")

# df_ground_truth = pd.DataFrame(ground_truth)
# print(f"\n✅ Loaded emotion labels: {len(df_ground_truth)} participants")
# print(f"\nGround truth summary:")
# print(f"  Valence: {df_ground_truth['true_valence'].mean():.2f} ± {df_ground_truth['true_valence'].std():.2f}")
# print(f"  Arousal: {df_ground_truth['true_arousal'].mean():.2f} ± {df_ground_truth['true_arousal'].std():.2f}")

# # Step 3: Merge Features with Ground Truth
# print("\n" + "="*70)
# print("STEP 3: Merging Features with Ground Truth")
# print("="*70)

# df_complete = df_features.merge(df_ground_truth, on='participant_id', how='inner')
# print(f"\n✅ Complete dataset: {len(df_complete)} participants with both features and labels")

# df_complete.to_csv("kemocon_complete_data.csv", index=False)
# print("✅ Saved: kemocon_complete_data.csv")

# # Step 4: Run LLM Inference
# print("\n" + "="*70)
# print("STEP 4: Running LLM Inference")
# print("="*70)

# def call_llm_emotion_inference(row):
#     """Call Groq LLM for emotion inference (same as Fitbit pipeline)"""
    
#     prompt = f"""You are an expert in affective computing and physiological emotion recognition.

# Given the following physiological data from a person, infer their emotional state in terms of VALENCE and AROUSAL.

# EMOTION DIMENSIONS:
# - VALENCE: How positive/negative the emotion feels (1=very negative, 5=very positive)
# - AROUSAL: How activated/calm the person feels (1=very calm, 5=very activated)

# PHYSIOLOGICAL DATA:
# - Mean Heart Rate: {row['hr_mean']:.1f} bpm
# - Heart Rate Std Dev: {row['hr_std']:.1f} bpm
# - Min Heart Rate: {row['hr_min']:.1f} bpm
# - Max Heart Rate: {row['hr_max']:.1f} bpm
# - HRV (RMSSD): {row['hrv_rmssd']:.1f} ms

# CONTEXT:
# - Higher HR usually indicates higher arousal
# - Lower HRV (RMSSD) often indicates stress (negative valence)
# - Higher HRV indicates relaxation (positive valence)

# OUTPUT FORMAT (CRITICAL):
# Respond with ONLY two numbers separated by a comma: valence,arousal
# Example: 3,4

# Your response:"""

#     try:
#         response = requests.post(
#             "https://api.groq.com/openai/v1/chat/completions",
#             headers={
#                 "Authorization": f"Bearer {GROQ_API_KEY}",
#                 "Content-Type": "application/json"
#             },
#             json={
#                 "model": MODEL,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.1,
#                 "max_tokens": 50
#             }
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             llm_response = result['choices'][0]['message']['content'].strip()
            
#             # Parse response
#             parts = llm_response.split(',')
#             if len(parts) == 2:
#                 valence = float(parts[0].strip())
#                 arousal = float(parts[1].strip())
#                 return valence, arousal, llm_response
#             else:
#                 return None, None, f"PARSE_ERROR: {llm_response}"
#         else:
#             return None, None, f"API_ERROR: {response.status_code}"
            
#     except Exception as e:
#         return None, None, f"ERROR: {str(e)}"

# # Run inference
# print(f"\nRunning LLM inference on {len(df_complete)} participants...")
# print("(This may take a few minutes due to API rate limits)")

# results = []

# for idx, row in df_complete.iterrows():
#     print(f"\n  [{idx+1}/{len(df_complete)}] P{row['participant_id']}: ", end='')
    
#     valence, arousal, response = call_llm_emotion_inference(row)
    
#     if valence is not None:
#         print(f"Predicted V={valence:.1f}, A={arousal:.1f} | True V={row['true_valence']:.1f}, A={row['true_arousal']:.1f}")
#     else:
#         print(f"Error: {response}")
    
#     results.append({
#         'participant_id': row['participant_id'],
#         'hr_mean': row['hr_mean'],
#         'hr_std': row['hr_std'],
#         'hrv_rmssd': row['hrv_rmssd'],
#         'true_valence': row['true_valence'],
#         'true_arousal': row['true_arousal'],
#         'pred_valence': valence,
#         'pred_arousal': arousal,
#         'llm_response': response
#     })
    
#     # Rate limiting
#     time.sleep(2)

# # Save results
# df_results = pd.DataFrame(results)
# df_results.to_csv("kemocon_llm_predictions.csv", index=False)

# print("\n" + "="*70)
# print("✅ INFERENCE COMPLETE")
# print("="*70)
# print(f"\n✅ Saved: kemocon_llm_predictions.csv")

# # Step 5: Calculate Validation Metrics
# print("\n" + "="*70)
# print("STEP 5: Validation Metrics (LLM vs Ground Truth)")
# print("="*70)

# # Remove failed predictions
# df_valid = df_results.dropna(subset=['pred_valence', 'pred_arousal'])

# print(f"\nValid predictions: {len(df_valid)}/{len(df_results)}")

# if len(df_valid) > 0:
#     from sklearn.metrics import mean_absolute_error, r2_score
#     from scipy.stats import pearsonr
    
#     # Valence metrics
#     mae_v = mean_absolute_error(df_valid['true_valence'], df_valid['pred_valence'])
#     r2_v = r2_score(df_valid['true_valence'], df_valid['pred_valence'])
#     corr_v, p_v = pearsonr(df_valid['true_valence'], df_valid['pred_valence'])
    
#     # Arousal metrics
#     mae_a = mean_absolute_error(df_valid['true_arousal'], df_valid['pred_arousal'])
#     r2_a = r2_score(df_valid['true_arousal'], df_valid['pred_arousal'])
#     corr_a, p_a = pearsonr(df_valid['true_arousal'], df_valid['pred_arousal'])
    
#     print("\nVALENCE:")
#     print(f"  MAE:        {mae_v:.3f}")
#     print(f"  R²:         {r2_v:.3f}")
#     print(f"  Pearson r:  {corr_v:.3f} (p={p_v:.4f})")
    
#     print("\nAROUSAL:")
#     print(f"  MAE:        {mae_a:.3f}")
#     print(f"  R²:         {r2_a:.3f}")
#     print(f"  Pearson r:  {corr_a:.3f} (p={p_a:.4f})")
    
#     # Save metrics
#     metrics = pd.DataFrame([
#         {'Metric': 'MAE', 'Valence': mae_v, 'Arousal': mae_a},
#         {'Metric': 'R²', 'Valence': r2_v, 'Arousal': r2_a},
#         {'Metric': 'Pearson r', 'Valence': corr_v, 'Arousal': corr_a},
#         {'Metric': 'p-value', 'Valence': p_v, 'Arousal': p_a}
#     ])
    
#     metrics.to_csv("kemocon_validation_metrics.csv", index=False)
#     print("\n✅ Saved: kemocon_validation_metrics.csv")

# print("\n" + "="*70)
# print("✅ K-EMOCON VALIDATION COMPLETE!")
# print("="*70)
# print("\nGenerated files:")
# print("  1. kemocon_complete_data.csv - Features + ground truth")
# print("  2. kemocon_llm_predictions.csv - LLM predictions")
# print("  3. kemocon_validation_metrics.csv - Validation metrics")



"""
K-EmoCon LLM Inference - FIXED VERSION
Stricter prompt to force correct output format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time

print("="*70)
print("K-EMOCON LLM INFERENCE - FIXED VERSION")
print("="*70)

# Configuration
extract_dir = Path("K-EmoCon/extracted")
e4_data_dir = extract_dir / "e4_data"
annotation_dir = extract_dir / "emotion_annotations" / "aggregated_external_annotations"

GROQ_API_KEY = "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"
MODEL = "llama-3.3-70b-versatile"

# Load pre-computed data
df_complete = pd.read_csv("kemocon_complete_data.csv")
print(f"\n✅ Loaded: {len(df_complete)} participants")

# Check for existing results
try:
    df_existing = pd.read_csv("kemocon_llm_predictions.csv")
    existing_ids = set(df_existing[df_existing['pred_valence'].notna()]['participant_id'].astype(str))
    print(f"✅ Found existing results: {len(existing_ids)} completed")
except:
    existing_ids = set()
    print("No existing results found - starting fresh")

def call_llm_emotion_inference(row):
    """Call Groq LLM with STRICT output format"""
    
    # FIXED PROMPT - much more explicit
    prompt = f"""Based on this heart rate data, predict emotion scores.

Heart Rate: {row['hr_mean']:.1f} bpm (std={row['hr_std']:.1f}, min={row['hr_min']:.1f}, max={row['hr_max']:.1f})
HRV RMSSD: {row['hrv_rmssd']:.1f} ms

Return ONLY two numbers (1-5 scale):
- First number: Valence (1=negative, 5=positive)
- Second number: Arousal (1=calm, 5=excited)

Format: X,Y
Example: 3,4

Your answer (numbers only):"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a data labeling assistant. Respond ONLY with the requested format. No explanations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 10  # Force short response
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content'].strip()
            
            # Clean response - remove any text, keep only numbers and comma
            import re
            numbers = re.findall(r'\d+\.?\d*', llm_response)
            
            if len(numbers) >= 2:
                valence = float(numbers[0])
                arousal = float(numbers[1])
                
                # Clamp to 1-5 range
                valence = max(1, min(5, valence))
                arousal = max(1, min(5, arousal))
                
                return valence, arousal, llm_response
            else:
                return None, None, f"PARSE_ERROR: {llm_response}"
        else:
            return None, None, f"API_ERROR: {response.status_code}"
            
    except Exception as e:
        return None, None, f"ERROR: {str(e)}"

# Run inference
print("\n" + "="*70)
print("RUNNING LLM INFERENCE")
print("="*70)
print(f"\nProcessing {len(df_complete)} participants...")

results = []
success_count = 0
error_count = 0

for idx, row in df_complete.iterrows():
    participant_id = str(row['participant_id'])
    
    # Skip if already completed
    if participant_id in existing_ids:
        print(f"  [{idx+1}/{len(df_complete)}] P{participant_id}: ✅ Already completed")
        continue
    
    print(f"  [{idx+1}/{len(df_complete)}] P{participant_id}: ", end='', flush=True)
    
    valence, arousal, response = call_llm_emotion_inference(row)
    
    if valence is not None:
        error_v = abs(valence - row['true_valence'])
        error_a = abs(arousal - row['true_arousal'])
        print(f"V={valence:.1f} A={arousal:.1f} | True: V={row['true_valence']:.1f} A={row['true_arousal']:.1f} | Error: V={error_v:.1f} A={error_a:.1f}")
        success_count += 1
    else:
        print(f"❌ {response[:50]}")
        error_count += 1
    
    results.append({
        'participant_id': participant_id,
        'hr_mean': row['hr_mean'],
        'hr_std': row['hr_std'],
        'hrv_rmssd': row['hrv_rmssd'],
        'true_valence': row['true_valence'],
        'true_arousal': row['true_arousal'],
        'pred_valence': valence,
        'pred_arousal': arousal,
        'llm_response': response
    })
    
    # Rate limiting
    time.sleep(2)

# Save results
df_results = pd.DataFrame(results)

# Merge with existing if any
try:
    df_existing = pd.read_csv("kemocon_llm_predictions.csv")
    df_results = pd.concat([df_existing, df_results], ignore_index=True)
    df_results = df_results.drop_duplicates(subset=['participant_id'], keep='last')
except:
    pass

df_results.to_csv("kemocon_llm_predictions.csv", index=False)

print("\n" + "="*70)
print("INFERENCE COMPLETE")
print("="*70)
print(f"\n✅ Success: {success_count}")
print(f"❌ Errors: {error_count}")
print(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")

# Calculate Validation Metrics
print("\n" + "="*70)
print("VALIDATION METRICS")
print("="*70)

df_valid = df_results.dropna(subset=['pred_valence', 'pred_arousal'])

print(f"\nValid predictions: {len(df_valid)}/{len(df_results)}")

if len(df_valid) >= 5:
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    
    mae_v = mean_absolute_error(df_valid['true_valence'], df_valid['pred_valence'])
    r2_v = r2_score(df_valid['true_valence'], df_valid['pred_valence'])
    corr_v, p_v = pearsonr(df_valid['true_valence'], df_valid['pred_valence'])
    
    mae_a = mean_absolute_error(df_valid['true_arousal'], df_valid['pred_arousal'])
    r2_a = r2_score(df_valid['true_arousal'], df_valid['pred_arousal'])
    corr_a, p_a = pearsonr(df_valid['true_arousal'], df_valid['pred_arousal'])
    
    print("\nVALENCE:")
    print(f"  MAE:        {mae_v:.3f}")
    print(f"  R²:         {r2_v:.3f}")
    print(f"  Pearson r:  {corr_v:.3f} (p={p_v:.4f})")
    
    print("\nAROUSAL:")
    print(f"  MAE:        {mae_a:.3f}")
    print(f"  R²:         {r2_a:.3f}")
    print(f"  Pearson r:  {corr_a:.3f} (p={p_a:.4f})")
    
    metrics = pd.DataFrame([
        {'Metric': 'MAE', 'Valence': mae_v, 'Arousal': mae_a},
        {'Metric': 'R²', 'Valence': r2_v, 'Arousal': r2_a},
        {'Metric': 'Pearson r', 'Valence': corr_v, 'Arousal': corr_a}
    ])
    
    metrics.to_csv("kemocon_validation_metrics.csv", index=False)
    print("\n✅ Saved: kemocon_validation_metrics.csv")
else:
    print("\n⚠️  Not enough valid predictions for metrics")

print("\n" + "="*70)
print("FILES GENERATED:")
print("="*70)
print("  1. kemocon_llm_predictions.csv")
print("  2. kemocon_validation_metrics.csv")