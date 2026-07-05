"""
extract_hr_signals.py
=====================
INNER DIRECTORY: InHouseCollectedData/

Extracts Fitbit heart rate signals aligned to clip-viewing timestamps.
Window direction: [created_at - WINDOW_SEC, created_at]
  → created_at is when the user SUBMITTED their rating (clip just ended)
  → so we look BACKWARDS to capture physiology DURING the clip

Follows Health-LLM (Kim et al., 2024):
  - Temporal context: raw HR sequence
  - Statistical summary: mean, std, min, max, drift
  - HRV: RMSSD computed from successive HR differences

Outputs: fitbit_hr_context.csv
"""

import pandas as pd
import json
import os
import glob
from datetime import datetime, timedelta
import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
WINDOW_SEC = 300        # 5-minute backward window from clip submission
                        # Increase to 600 if you get fewer than 3 samples per clip

MASTER_DATA_FILE = 'clip_responses_export.xlsx'   # in same (inner) directory

# Map xlsx participant names → Fitbit data folder names
# (folder names are the subfolders inside InHouseCollectedData/)
PARTICIPANT_MAP = {
    'Nishant':       'Nishant',
    'debjit2001':    'Debjit',
    'pranjal123':    'pranjal',
    'aritra':        'aritra',
    'sayantankuila': 'sayantan',
    'ajaycc17':      'Ajay',
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def compute_rmssd(bpm_list):
    """
    RMSSD from BPM values.
    Converts BPM → RR intervals (ms), then computes root-mean-square
    of successive differences. Health-LLM uses RMSSD as HRV feature.
    """
    if len(bpm_list) < 2:
        return np.nan
    bpm_arr = np.array(bpm_list, dtype=float)
    rr_ms = 60000.0 / bpm_arr          # BPM → RR interval in milliseconds
    successive_diffs = np.diff(rr_ms)
    return float(np.sqrt(np.mean(successive_diffs ** 2)))


def find_hr_file(folder_name, date_str):
    """
    Recursively search for heart_rate-YYYY-MM-DD.json inside the
    participant's folder (which may contain nested takeout subfolders).
    """
    pattern = os.path.join('.', folder_name, '**', f'heart_rate-{date_str}.json')
    files = glob.glob(pattern, recursive=True)
    return files[0] if files else None


def extract_bpm_window(file_path, end_dt, window_sec=WINDOW_SEC):
    """
    Extract BPM readings in the window [end_dt - window_sec, end_dt].
    end_dt = created_at (the moment the user submitted their rating,
             i.e., the clip had JUST finished).

    Fitbit heart_rate JSON format:
      [{"dateTime": "MM/DD/YY HH:MM:SS", "value": {"bpm": 72, "confidence": 2}}, ...]
    """
    start_dt = end_dt - timedelta(seconds=window_sec)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return []

    readings = []
    for entry in data:
        try:
            entry_time = datetime.strptime(entry['dateTime'], '%m/%d/%y %H:%M:%S')
            value = entry.get('value', {})

            if isinstance(value, dict):
                bpm = value.get('bpm') or value.get('resting')
            elif isinstance(value, (int, float)):
                bpm = float(value)
            else:
                continue

            if bpm and start_dt <= entry_time <= end_dt:
                readings.append(float(bpm))
        except Exception:
            continue

    return readings


# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────

df_master = pd.read_excel(MASTER_DATA_FILE)

print(f"Loaded {len(df_master)} rows from {MASTER_DATA_FILE}")
print(f"Participants in xlsx: {df_master['participant'].unique().tolist()}")
print(f"Window: {WINDOW_SEC}s before clip submission\n")

results = []
skipped_no_file = 0
skipped_no_data = 0

for _, row in df_master.iterrows():
    participant = row['participant']

    if participant not in PARTICIPANT_MAP:
        continue

    folder_name = PARTICIPANT_MAP[participant]
    timestamp   = row['created_at']               # pandas Timestamp
    date_str    = timestamp.strftime('%Y-%m-%d')  # YYYY-MM-DD for filename search

    hr_file = find_hr_file(folder_name, date_str)
    if not hr_file:
        skipped_no_file += 1
        continue

    bpm_readings = extract_bpm_window(hr_file, timestamp)

    if len(bpm_readings) < 2:
        # Not enough data points for statistics
        skipped_no_data += 1
        continue

    bpm_arr = np.array(bpm_readings, dtype=float)

    results.append({
        # Identifiers
        'participant':           participant,
        'clip_title':            row['clip_title'],
        'clip_order':            row['clip_order'],
        'timestamp':             timestamp,

        # Temporal context (Natural Language String, per Health-LLM §3.4)
        'hr_sequence':           bpm_readings,

        # Statistical summary (Health-LLM Statistical Summary method)
        'avg_bpm':               round(float(np.mean(bpm_arr)), 2),
        'hr_std':                round(float(np.std(bpm_arr)), 2),
        'hr_min':                round(float(np.min(bpm_arr)), 2),
        'hr_max':                round(float(np.max(bpm_arr)), 2),
        'bpm_drift':             round(float(bpm_arr[-1] - bpm_arr[0]), 2),
        'hr_count':              len(bpm_readings),

        # HRV feature (Health-LLM uses RMSSD for sleep disorder, we apply same)
        'hrv_rmssd':             round(compute_rmssd(bpm_readings), 2),

        # Ground truth labels from xlsx
        # impact_valence/arousal = what the clip INDUCED in the user (our prediction target)
        # participant_valence/arousal = how the user felt overall (secondary)
        'self_reported_valence': row['impact_valence'],
        'self_reported_arousal': row['impact_arousal'],
        'participant_valence':   row['participant_valence'],
        'participant_arousal':   row['participant_arousal'],
    })

output_df = pd.DataFrame(results)
output_df.to_csv('fitbit_hr_context.csv', index=False)

print(f"✅ Extracted HR windows for {len(results)} clip responses")
print(f"   Participants covered: {output_df['participant'].nunique()}")
print(f"   Unique clips covered: {output_df['clip_title'].nunique()}")
print(f"   Skipped (no HR file found): {skipped_no_file}")
print(f"   Skipped (< 2 readings in window): {skipped_no_data}")
print(f"\nOutput: fitbit_hr_context.csv")

if len(results) == 0:
    print("\n⚠️  No data extracted. Checklist:")
    print(f"   1. Is this script run FROM InHouseCollectedData/ directory?")
    print(f"   2. Do subfolders (Nishant/, Debjit/ etc.) exist here?")
    print(f"   3. Are Fitbit JSON files inside those subfolders?")
    print(f"   4. Do the file dates match clip timestamps in the xlsx?")
    print(f"   5. Try increasing WINDOW_SEC to 600 or 900")
