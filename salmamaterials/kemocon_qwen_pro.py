"""
kemocon_qwen_pro.py
===================
TASK 1: Improved Clinical Validation for Qwen-32B
- Integrates EDA (Electrodermal Activity) from E4 sensors.
- Implements Stratified Few-Shot (Low/High/Neutral anchors).
- Targets Arousal MAE < 0.3.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import time
from groq import Groq
from scipy.stats import pearsonr

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
EXTRACT_DIR    = Path("K-EmoCon/extracted")
E4_DATA_DIR    = EXTRACT_DIR / "e4_data"
ANNOTATION_DIR = EXTRACT_DIR / "emotion_annotations" / "aggregated_external_annotations"

# Re-using the same Groq key from previous context
GROQ_API_KEY   = "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"
MODEL_NAME     = "qwen-32b-preview" # Using Qwen-32B as requested
TEMPERATURE    = 0.0
MAX_TOKENS     = 60
HR_SEQ_LEN     = 30

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def compute_rmssd(ibi_values_sec):
    ibi_ms = np.array(ibi_values_sec, dtype=float) * 1000.0
    if len(ibi_ms) < 2: return np.nan
    diffs = np.diff(ibi_ms)
    return float(np.sqrt(np.mean(diffs ** 2)))

def load_e4_csv(file_path, value_col_index=0):
    try:
        df = pd.read_csv(file_path)
        if 'value' in df.columns:
            return pd.to_numeric(df['value'], errors='coerce').dropna().values
    except: pass
    df = pd.read_csv(file_path, header=None)
    return pd.to_numeric(df.iloc[2:, value_col_index], errors='coerce').dropna().values

def normalize_pid(raw_id):
    import re
    match = re.search(r'\d+', str(raw_id))
    return int(match.group(0)) if match else None

# ─── STEP 1/2: FEATURE EXTRACTION (including EDA) ─────────────────────────────

print("Starting Step 1 & 2: Multi-Modal Feature Extraction...")

# Load Annotations
all_anns = []
for f in ANNOTATION_DIR.glob("*.csv"):
    df = pd.read_csv(f)
    df['_pid'] = normalize_pid(f.stem)
    all_anns.append(df)
ann_df = pd.concat(all_anns)
# Detect Valence/Arousal columns
v_col = next(c for c in ann_df.columns if 'valence' in c.lower())
a_col = next(c for c in ann_df.columns if 'arousal' in c.lower())
annotation_map = ann_df.groupby('_pid').agg({v_col: 'mean', a_col: 'mean'}).to_dict('index')

feature_records = []
for p_dir in sorted(E4_DATA_DIR.iterdir(), key=lambda d: normalize_pid(d.name) or 999):
    if not p_dir.is_dir(): continue
    pid = normalize_pid(p_dir.name)
    
    # Load Sensors
    hr_vals = load_e4_csv(next(p_dir.rglob("E4_HR.csv")))
    ibi_vals = load_e4_csv(next(p_dir.rglob("E4_IBI.csv")), value_col_index=1)
    eda_vals = load_e4_csv(next(p_dir.rglob("E4_EDA.csv")))
    
    if len(hr_vals) == 0: continue
    
    # Feature Engineering
    rmssd = compute_rmssd(ibi_vals)
    # EDA Features (The Arousal Boost)
    eda_mean = round(float(np.mean(eda_vals)), 3) if len(eda_vals) > 0 else 0.0
    eda_range = round(float(np.max(eda_vals) - np.min(eda_vals)), 3) if len(eda_vals) > 0 else 0.0
    
    ann = annotation_map.get(pid, {})
    record = {
        'pid': pid,
        'hr_mean': round(float(np.mean(hr_vals)), 2),
        'hr_std': round(float(np.std(hr_vals)), 2),
        'hr_drift': round(float(hr_vals[-1] - hr_vals[0]), 2),
        'rmssd': round(rmssd, 2) if not np.isnan(rmssd) else 'N/A',
        'eda_mean': eda_mean,
        'eda_range': eda_range,
        'hr_sequence': hr_vals[:15].tolist() + ['...'] + hr_vals[-15:].tolist() if len(hr_vals) > 15 else hr_vals.tolist(),
        'true_v': round(ann.get(v_col, 0), 2),
        'true_a': round(ann.get(a_col, 0), 2)
    }
    feature_records.append(record)

features_df = pd.DataFrame(feature_records)

# ─── STEP 3: PROMPT CREATION (Stratified Few-Shot) ────────────────────────────

def create_pro_prompt(row, all_df):
    # Select Stratified Shots to anchor the 1-5 scale
    def get_shot(df, target_score_range):
        # target_score_range is a tuple like (1, 2)
        cand = df[(df.true_a >= target_score_range[0]) & (df.true_a <= target_score_range[1]) & (df.pid != row['pid'])]
        return cand.sample(1, random_state=42).iloc[0] if not cand.empty else None

    shots = [
        get_shot(all_df, (1, 2)), # Low Arousal Shot
        get_shot(all_df, (3, 3)), # Neutral Shot
        get_shot(all_df, (4, 5))  # High Arousal Shot
    ]
    
    fs_text = "### Calibration Examples (Clinical Ground Truth):\n"
    for s in shots:
        if s is not None:
            fs_text += f"- Input: [HR: {s['hr_mean']}, HRV: {s['rmssd']}, EDA Mean: {s['eda_mean']}, EDA Range: {s['eda_range']}] | Output: {{\"valence\": {s['true_v']}, \"arousal\": {s['true_a']}}}\n"

    prompt = f"""
### Instruction: Act as a high-precision Biological Proxy.
You must infer Valence (1: Sad, 5: Happy) and Arousal (1: Calm, 5: Excited) from E4 sensor data.

### Features Guide:
1. HRV (RMSSD): Higher usually correlates with stable/positive valence.
2. EDA (Electrodermal Activity): This is your PRIMARY signal for Arousal. High EDA Mean/Range = High Arousal.
3. HR Drift: Positive drift indicates increasing engagement.

{fs_text}

### CURRENT TASK:
- HR Sequence: {row['hr_sequence']}
- Statistics: Mean HR={row['hr_mean']}, RMSSD={row['rmssd']}, EDA Mean={row['eda_mean']} uS, EDA Range={row['eda_range']} uS.

Return ONLY a JSON response: {{"valence": X, "arousal": Y}}
"""
    return prompt.strip()

features_df['prompt'] = features_df.apply(lambda r: create_pro_prompt(r, features_df), axis=1)

# ─── STEP 4: INFERENCE (Qwen-32B) ─────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)
results = []

print(f"Running Inference on {len(features_df)} clinical sessions using {MODEL_NAME}...")

for i, row in features_df.iterrows():
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": row['prompt']}],
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            temperature=TEMPERATURE,
        )
        res = json.loads(completion.choices[0].message.content)
        results.append({
            'pid': row['pid'],
            'true_v': row['true_v'], 'pred_v': res.get('valence'),
            'true_a': row['true_a'], 'pred_a': res.get('arousal')
        })
        print(f"P{row['pid']}: Done.")
        time.sleep(1)
    except Exception as e:
        print(f"Error on P{row['pid']}: {e}")

res_df = pd.DataFrame(results)
res_df.to_csv("kemocon_qwen_pro_results.csv", index=False)

# ─── STEP 5: EVALUATION ───────────────────────────────────────────────────────

res_df = res_df.dropna()
mae_v = np.mean(np.abs(res_df.true_v - res_df.pred_v))
mae_a = np.mean(np.abs(res_df.true_a - res_df.pred_a))
pearson_v, _ = pearsonr(res_df.true_v, res_df.pred_v)
pearson_a, _ = pearsonr(res_df.true_a, res_df.pred_a)

metrics = {
    'model': 'Qwen-32B-Pro', 'n': len(res_df),
    'mae_v': round(float(mae_v), 3), 'mae_a': round(float(mae_a), 3),
    'r_v': round(float(pearson_v), 3), 'r_a': round(float(pearson_a), 3)
}
pd.DataFrame([metrics]).to_csv("kemocon_qwen_metrics.csv", index=False)
print("\n=== CLINICAL VALIDATION COMPLETE (QWEN PRO) ===")
print(f"Valence MAE: {mae_v:.3f} | Arousal MAE: {mae_a:.3f}")
