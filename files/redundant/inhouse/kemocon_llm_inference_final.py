"""
kemocon_llm_inference_final.py
===============================
OUTER DIRECTORY: (same level as K-EmoCon/ and InHouseCollectedData/)

Complete K-EmoCon validation pipeline following Health-LLM (Kim et al., 2024).

K-EmoCon dataset context:
  - 32 participants watched debate video clips
  - E4 wristband: HR at 1 Hz, IBI data
  - External observers annotated valence/arousal per participant per segment
  - K-EmoCon paper's key claim: physiologically-measured induced emotion
    agrees with self-reported labels → ideal for cross-dataset validation

This script:
  1. Loads E4 HR and IBI data per participant
  2. Extracts features (same set as in-house Fitbit pipeline)
  3. Creates Health-LLM 'All' context prompts (same format as in-house)
  4. Runs Llama and Mistral via Groq API
  5. Compares predictions against K-EmoCon annotation ground truth
  6. Outputs validation metrics + fine-tuning decision guidance

Health-LLM evaluation: MAE (primary), Pearson r, Accuracy ±1
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

GROQ_API_KEY   = "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"
TEMPERATURE    = 0.0
MAX_TOKENS     = 60

# How many HR values to include in the prompt sequence
# Health-LLM uses NaN, 991.0, ..., NaN style (first + last few values)
HR_SEQ_LEN     = 30

print("=" * 70)
print("K-EMOCON LLM VALIDATION PIPELINE")
print("Following Health-LLM (Kim et al., 2024)")
print("=" * 70)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def compute_rmssd(ibi_values_sec):
    """
    RMSSD from IBI values in seconds.
    Converts to ms first (standard HRV convention).
    """
    ibi_ms = np.array(ibi_values_sec, dtype=float) * 1000.0
    if len(ibi_ms) < 2:
        return np.nan
    diffs = np.diff(ibi_ms)
    return float(np.sqrt(np.mean(diffs ** 2)))


def load_e4_hr(hr_file):
    """
    Load E4 HR CSV.
    Row 0: Unix start timestamp
    Row 1: Sample rate (Hz), typically 1.0 for HR
    Rows 2+: HR values (BPM)
    Returns DataFrame with columns [timestamp_unix, hr]
    """
    df = pd.read_csv(hr_file, header=None)
    start_ts    = float(df.iloc[0, 0])
    sample_rate = float(df.iloc[1, 0])        # Hz
    hr_values   = df.iloc[2:, 0].astype(float).values

    n          = len(hr_values)
    timestamps = [start_ts + i / sample_rate for i in range(n)]

    return pd.DataFrame({'timestamp_unix': timestamps, 'hr': hr_values})


def load_e4_ibi(ibi_file):
    """
    Load E4 IBI CSV.
    Row 0: Unix start timestamp
    Rows 1+: (time_offset_seconds, IBI_seconds)
    Returns DataFrame with IBI values in seconds.
    """
    df = pd.read_csv(ibi_file, header=None)
    if len(df) < 2:
        return pd.DataFrame()

    start_ts = float(df.iloc[0, 0])
    ibi_data = df.iloc[1:].copy()
    ibi_data.columns = ['time_offset_sec', 'ibi_sec']
    ibi_data = ibi_data.astype(float)
    ibi_data['timestamp_unix'] = start_ts + ibi_data['time_offset_sec']
    return ibi_data.reset_index(drop=True)


def detect_annotation_columns(df):
    """Flexibly detect valence/arousal column names in annotation files."""
    valence_col = next((c for c in df.columns if 'valence' in c.lower()), None)
    arousal_col = next((c for c in df.columns if 'arousal' in c.lower()), None)
    return valence_col, arousal_col


def normalize_participant_id(raw_id):
    """Normalize participant IDs like 'P01', '1', 'p1' → integer."""
    try:
        return int(str(raw_id).upper().replace('P', '').strip())
    except ValueError:
        return None


# ─── STEP 1: LOAD ANNOTATIONS ─────────────────────────────────────────────────

print(f"\n{'─'*70}")
print("Step 1: Loading K-EmoCon emotion annotations (ground truth)")
print(f"{'─'*70}")

if not ANNOTATION_DIR.exists():
    print(f"⚠️  Annotation directory not found: {ANNOTATION_DIR}")
    print("   Have you extracted emotion_annotations.tar.gz into K-EmoCon/extracted/?")
    print("   Run kemocon_validation_complete.py first to extract the archives.")
    annotation_map = {}
    VALENCE_COL = AROUSAL_COL = None
else:
    annotation_files = list(ANNOTATION_DIR.glob("*.csv"))
    print(f"Found {len(annotation_files)} annotation files")

    all_anns = []
    for f in annotation_files:
        try:
            df = pd.read_csv(f)
            # Detect participant ID from filename (e.g., "P01.csv" → 1)
            pid = normalize_participant_id(f.stem)
            df['_participant_int'] = pid
            all_anns.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if all_anns:
        ann_df = pd.concat(all_anns, ignore_index=True)
        VALENCE_COL, AROUSAL_COL = detect_annotation_columns(ann_df)
        print(f"Annotation columns found: {ann_df.columns.tolist()}")
        print(f"Valence column: {VALENCE_COL}")
        print(f"Arousal column: {AROUSAL_COL}")
        print(f"Total annotation rows: {len(ann_df)}")

        # Build per-participant average label map
        annotation_map = {}
        if VALENCE_COL and AROUSAL_COL:
            for pid, group in ann_df.groupby('_participant_int'):
                annotation_map[pid] = {
                    'true_valence': round(group[VALENCE_COL].mean(), 3),
                    'true_arousal': round(group[AROUSAL_COL].mean(), 3),
                    'n_annotations': len(group),
                }
        print(f"Built annotation map for {len(annotation_map)} participants")
    else:
        ann_df = pd.DataFrame()
        annotation_map = {}
        VALENCE_COL = AROUSAL_COL = None
        print("WARNING: No annotation data loaded")


# ─── STEP 2: EXTRACT HR FEATURES ──────────────────────────────────────────────

print(f"\n{'─'*70}")
print("Step 2: Extracting HR and HRV features from E4 data")
print(f"{'─'*70}")

if not E4_DATA_DIR.exists():
    print(f"⚠️  E4 data directory not found: {E4_DATA_DIR}")
    print("   Have you extracted e4_data.tar into K-EmoCon/extracted/?")
    feature_records = []
else:
    feature_records = []
    participant_dirs = sorted(
        [d for d in E4_DATA_DIR.iterdir() if d.is_dir()],
        key=lambda d: normalize_participant_id(d.name) or 999
    )

    print(f"Found {len(participant_dirs)} participant directories")

    for p_dir in participant_dirs:
        participant_raw = p_dir.name
        participant_int = normalize_participant_id(participant_raw)

        # Find HR file (search recursively for E4_HR.csv)
        hr_files = list(p_dir.rglob("E4_HR.csv"))
        if not hr_files:
            print(f"  {participant_raw}: No E4_HR.csv found — skipping")
            continue

        hr_file = hr_files[0]

        try:
            hr_df = load_e4_hr(hr_file)
            hr_vals = hr_df['hr'].values

            if len(hr_vals) == 0:
                continue

            # RMSSD from IBI data
            rmssd = np.nan
            ibi_files = list(p_dir.rglob("E4_IBI.csv"))
            if ibi_files:
                try:
                    ibi_df = load_e4_ibi(ibi_files[0])
                    if len(ibi_df) >= 2:
                        rmssd = compute_rmssd(ibi_df['ibi_sec'].values)
                except Exception:
                    pass

            # Natural Language String representation (Health-LLM §3.4)
            # Use first HR_SEQ_LEN values; if longer, show first few + last few
            if len(hr_vals) <= HR_SEQ_LEN:
                hr_seq_list = hr_vals.tolist()
            else:
                # Show first 15, NaN gap, last 15 — matches Health-LLM format
                first_part = hr_vals[:15].tolist()
                last_part  = hr_vals[-15:].tolist()
                hr_seq_list = first_part + ['...'] + last_part

            # Get annotation
            ann = annotation_map.get(participant_int, {})

            record = {
                'participant_id':      participant_raw,
                'participant_int':     participant_int,
                'hr_mean':             round(float(np.mean(hr_vals)), 2),
                'hr_std':              round(float(np.std(hr_vals)), 2),
                'hr_min':              round(float(np.min(hr_vals)), 2),
                'hr_max':              round(float(np.max(hr_vals)), 2),
                'hr_drift':            round(float(hr_vals[-1] - hr_vals[0]), 2),
                'hrv_rmssd':           round(rmssd, 2) if not np.isnan(rmssd) else 'N/A',
                'hr_sequence':         hr_seq_list,
                'n_hr_samples':        len(hr_vals),
                'true_valence':        ann.get('true_valence', np.nan),
                'true_arousal':        ann.get('true_arousal', np.nan),
                'n_annotations':       ann.get('n_annotations', 0),
            }
            feature_records.append(record)

            ann_str = (f"V={ann.get('true_valence', '?'):.2f} A={ann.get('true_arousal', '?'):.2f}"
                       if ann else "no annotation")
            print(f"  {participant_raw:6s}: HR={record['hr_mean']:.1f}±{record['hr_std']:.1f}  "
                  f"RMSSD={record['hrv_rmssd']}  {ann_str}")

        except Exception as e:
            print(f"  {participant_raw}: ERROR — {e}")

print(f"\nExtracted features for {len(feature_records)} participants")

features_df = pd.DataFrame(feature_records)
if len(features_df) > 0:
    features_df.to_csv('kemocon_features.csv', index=False)
    valid_labels = features_df.dropna(subset=['true_valence', 'true_arousal'])
    print(f"Participants with valid annotation labels: {len(valid_labels)}")
else:
    print("⚠️  No features extracted. Check K-EmoCon directory structure.")
    import sys; sys.exit(1)


# ─── STEP 3: CREATE PROMPTS (Health-LLM 'All' context) ────────────────────────

print(f"\n{'─'*70}")
print("Step 3: Creating prompts (Health-LLM 'All' context strategy)")
print(f"{'─'*70}")


def create_kemocon_prompt(row):
    """
    Health-LLM 'All' context: Health Knowledge + Temporal Sequence + Question
    No User Context here because K-EmoCon doesn't provide detailed demographics
    in a format matching our pipeline. We include what is available.
    """
    # ── Health Context (hc) ──────────────────────────────────────────────────
    health_ctx = (
        "Valence refers to the subjective emotional positivity or negativity, "
        "measured from 1 (very negative / sad) to 5 (very positive / happy). "
        "Arousal refers to physiological and psychological activation level, "
        "measured from 1 (very calm / relaxed) to 5 (very excited / activated). "
        "Higher arousal is typically associated with elevated heart rate. "
        "Heart rate variability (RMSSD) decreases under emotional stress and "
        "negative valence. Heart rate drift over a session reflects cumulative "
        "emotional engagement."
    )

    # ── Temporal Context (tc) — Natural Language String (Health-LLM §3.4) ───
    hr_seq = row['hr_sequence']
    if isinstance(hr_seq, str):
        hr_seq_str = hr_seq
    else:
        hr_seq_str = str(hr_seq)

    temporal_ctx = (
        f"[Heart Rate Sequence (BPM)]: {hr_seq_str}. "
        f"[Statistical Summary]: "
        f"Mean = {row['hr_mean']} BPM, "
        f"Std = {row['hr_std']} BPM, "
        f"Min = {row['hr_min']} BPM, "
        f"Max = {row['hr_max']} BPM, "
        f"Drift (last − first) = {row['hr_drift']} BPM, "
        f"HRV RMSSD = {row['hrv_rmssd']} ms."
    )

    prompt = (
        f"### Instruction: You are an intelligent healthcare agent specializing in "
        f"affective computing and physiological signal analysis.\n\n"

        f"### Health Knowledge: {health_ctx}\n\n"

        f"### Activity: The participant is watching debate video clips in a "
        f"controlled laboratory setting.\n\n"

        f"### Sensor Readings: {temporal_ctx}\n\n"

        f"### Question: Based on the physiological data above, predict the "
        f"participant's Valence and Arousal scores on a 1–5 integer scale.\n\n"

        f"### Response: Return ONLY a JSON object: "
        f'{{\"valence\": X, \"arousal\": Y}}'
    )

    return prompt.strip()


features_df['llm_question'] = features_df.apply(create_kemocon_prompt, axis=1)
features_df.to_csv('kemocon_features.csv', index=False)
print(f"Prompts created for {len(features_df)} participants")


# ─── STEP 4: LLM INFERENCE ────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print("Step 4: LLM Inference (Llama + Mistral)")
print(f"{'─'*70}")

client = Groq(api_key=GROQ_API_KEY)

# Only run on participants that have ground truth labels
valid_df = features_df.dropna(subset=['true_valence', 'true_arousal']).reset_index(drop=True)
print(f"Running inference on {len(valid_df)} participants (those with annotations)")


def run_inference_loop(df, model_name, output_file):
    """Run LLM inference with resume logic and retry handling."""

    # Resume logic
    if os.path.exists(output_file):
        start_row = len(pd.read_csv(output_file))
        print(f"\n  {model_name}: Resuming from row {start_row}...")
    else:
        start_row = 0
        header_cols = list(df.columns) + ['inferred_valence', 'inferred_arousal', 'raw_response']
        pd.DataFrame(columns=header_cols).to_csv(output_file, index=False)
        print(f"\n  {model_name}: Starting fresh...")

    print(f"  Model: {model_name}")

    for i in range(start_row, len(df)):
        row = df.iloc[i]
        print(f"  [{i+1:02d}/{len(df)}] P={row['participant_id']}",
              end="  ", flush=True)

        inf_v = inf_a = None
        raw   = ""
        retries = 0

        while retries < 5:
            try:
                completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": row['llm_question']}],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw = completion.choices[0].message.content.strip()
                res = json.loads(raw)
                inf_v = res.get('valence')
                inf_a = res.get('arousal')
                print(f"V={inf_v}  A={inf_a}  (true: V={row['true_valence']}  A={row['true_arousal']})")
                time.sleep(1.5)
                break

            except Exception as e:
                retries += 1
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = 45 * retries
                    print(f"\n    [Rate limit] Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"\n    [Error] {err}")
                    time.sleep(15)
                if retries >= 5:
                    raw = f"ERROR: {err}"
                    print(f"    [Skip]")

        new_row = list(row) + [inf_v, inf_a, raw]
        pd.DataFrame([new_row]).to_csv(output_file, mode='a', header=False, index=False)

    result_df = pd.read_csv(output_file)
    print(f"  → Saved {len(result_df)} rows to {output_file}")
    return result_df


llama_results   = run_inference_loop(valid_df, "llama-3.3-70b-versatile", "kemocon_llama_results.csv")
mistral_results = run_inference_loop(valid_df, "mixtral-8x7b-32768",       "kemocon_mistral_results.csv")


# ─── STEP 5: EVALUATION ───────────────────────────────────────────────────────

print(f"\n{'─'*70}")
print("Step 5: Evaluation vs K-EmoCon Ground Truth")
print(f"{'─'*70}")


def evaluate_kemocon(df, model_name):
    """MAE, Pearson r, Accuracy — Health-LLM Table 3 format."""
    df = df.copy()
    for col in ['inferred_valence', 'inferred_arousal', 'true_valence', 'true_arousal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['inferred_valence', 'inferred_arousal',
                            'true_valence', 'true_arousal'])
    n = len(df)
    if n < 3:
        print(f"\n{model_name}: n={n} too small for evaluation")
        return {}

    pv = df['inferred_valence'].values.astype(float)
    pa = df['inferred_arousal'].values.astype(float)
    tv = df['true_valence'].values.astype(float)
    ta = df['true_arousal'].values.astype(float)

    mae_v  = float(np.mean(np.abs(pv - tv)))
    mae_a  = float(np.mean(np.abs(pa - ta)))
    r_v, p_v = pearsonr(pv, tv)
    r_a, p_a = pearsonr(pa, ta)
    acc_v  = float(np.mean(np.abs(pv - tv) <= 1.0) * 100)
    acc_a  = float(np.mean(np.abs(pa - ta) <= 1.0) * 100)

    print(f"\n  Model: {model_name}   (n = {n})")
    print(f"  {'Metric':<25} {'Valence':>10} {'Arousal':>10}")
    print(f"  {'-'*47}")
    print(f"  {'MAE (↓)':<25} {mae_v:>10.3f} {mae_a:>10.3f}")
    print(f"  {'Pearson r (↑)':<25} {r_v:>10.3f} {r_a:>10.3f}")
    print(f"  {'p-value':<25} {p_v:>10.4f} {p_a:>10.4f}")
    print(f"  {'Accuracy ±1 (%)':<25} {acc_v:>9.1f}% {acc_a:>9.1f}%")

    return {
        'model':                 model_name,
        'dataset':               'K-EmoCon',
        'n':                     n,
        'mae_valence':           round(mae_v, 3),
        'mae_arousal':           round(mae_a, 3),
        'pearson_r_valence':     round(r_v, 3),
        'pearson_p_valence':     round(p_v, 4),
        'pearson_r_arousal':     round(r_a, 3),
        'pearson_p_arousal':     round(p_a, 4),
        'accuracy_v_within1':    round(acc_v, 1),
        'accuracy_a_within1':    round(acc_a, 1),
    }


eval_llama   = evaluate_kemocon(llama_results,   "Llama-3.3-70B")
eval_mistral = evaluate_kemocon(mistral_results, "Mixtral-8x7B")

metrics_list = [m for m in [eval_llama, eval_mistral] if m]

if metrics_list:
    validation_df = pd.DataFrame(metrics_list)
    validation_df.to_csv('kemocon_validation_metrics.csv', index=False)

    print(f"\n{'='*70}")
    print("  K-EMOCON VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Files generated:")
    print(f"    kemocon_features.csv           — HR features + true labels")
    print(f"    kemocon_llama_results.csv       — Llama predictions")
    print(f"    kemocon_mistral_results.csv     — Mistral predictions")
    print(f"    kemocon_validation_metrics.csv  — Final comparison table")

    # ── Fine-tuning decision guidance ────────────────────────────────────────
    min_mae = min(m['mae_valence'] for m in metrics_list)
    best_model = min(metrics_list, key=lambda m: m['mae_valence'])['model']

    print(f"\n  ── Fine-tuning Decision ─────────────────────────────────────────")
    print(f"  Best model on K-EmoCon: {best_model} (V_MAE = {min_mae:.3f})")

    if min_mae > 1.0:
        print(f"\n  ⚠️  MAE > 1.0 → Performance below acceptable threshold")
        print(f"  RECOMMENDATION: Fine-tune on 10% of K-EmoCon")
        print(f"\n  Fine-tuning approach (Health-LLM §5.5):")
        print(f"  - 10% training split → already outperforms zero-shot in paper")
        print(f"  - Use instruction fine-tuning (not LoRA for simplicity)")
        print(f"  - Same prompt format, add 'Response: {{\"valence\": X, \"arousal\": Y}}'")
        print(f"  - Fine-tune on {int(len(valid_df) * 0.1)} samples")
        print(f"  - Validate on remaining {int(len(valid_df) * 0.9)} samples")
        print(f"\n  For Groq/API fine-tuning, use Groq's fine-tuning endpoint:")
        print(f"  https://console.groq.com/docs/fine-tuning")
    else:
        print(f"\n  ✅ MAE ≤ 1.0 → Zero-shot performance is acceptable")
        print(f"  K-EmoCon cross-dataset validation PASSES")
        print(f"  This confirms the LLM generalizes beyond in-house data")
        print(f"\n  Best model to use: {best_model}")
