"""
_taskH_infer_I2_with_EDA.py
============================
Physiological Extraction + LLM Inference  (I2 valence & arousal)

The Empatica EmbracePlus device was worn continuously across ALL participant
sessions.  Its data is stored as date-split files in one folder tree.
Alignment is purely timestamp-based: for every clip row in the responses CSV,
we slice the Empatica stream at [timestamp_clip_start, timestamp_clip_end].
No per-user device mapping is needed.

Empatica data is at 1-minute resolution (aggregated_per_minute).
Rows where missing_value_reason is not NaN are excluded (watch not worn /
low signal / not recording).

Target: 210 rows  (clips whose time windows fall inside the Empatica stream).
Output: _validation_with_I2.csv  (210 rows with I2_val, I2_aro populated)

Run:
  set GROQ_API_KEY=gsk_...
  python _taskH_infer_I2_with_EDA.py

Requirements: pip install groq pandas numpy
"""

import os, json, time, glob, math
import numpy as np
import pandas as pd
from groq import Groq

# ─── CONFIGURATION  ────────────────────────────────────────────────────────

RESPONSES_CSV = r"D:\BTP\btp2main\clip_responses-export-2026-04-12_14-51-50.csv"
EMPATICA_DIR  = r"D:\BTP\btp2main\2lf92x651f5z6t6y"          # recursive scan
EXPRESSED_CSV = r"D:\BTP\btp2\salmamaterials\final_expressed_valence_AUDIO.csv"
OUTPUT_CSV    = r"D:\BTP\btp2\salmamaterials\_validation_with_I2.csv"

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
LLM_MODEL     = "llama-3.3-70b-versatile"

# Blocks that represent real stimuli (skip Filler and Baseline_Buffer)
VALID_BLOCKS  = {"Maintenance_Block", "Drift_Block",
                 "Hyp9_Maintenance_Block", "Hyp9_Transition_Block"}

MAX_RETRIES   = 3
RETRY_DELAY   = 2.0   # seconds between retries

# The Empatica rows are 1-minute buckets.  We extend the clip window by this
# many ms on the left so we include the minute that started just before the
# clip began (handles the case where t_start falls mid-minute).
WINDOW_PAD_MS = 60_000


# ─── EMPATICA LOADER  ──────────────────────────────────────────────────────

def load_signal(signal_suffix: str) -> pd.DataFrame:
    """
    Recursively find all CSV files ending in  *_{signal_suffix}.csv
    under EMPATICA_DIR, concatenate them, drop "watch not worn" rows,
    and return a clean DataFrame with columns:
        timestamp_unix  (int, ms UTC)
        value           (float)
    Sorted by timestamp_unix ascending.
    """
    pattern = os.path.join(EMPATICA_DIR, "**", f"*_{signal_suffix}.csv")
    files   = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"  [WARN] No *_{signal_suffix}.csv files found under {EMPATICA_DIR}")
        return pd.DataFrame(columns=["timestamp_unix", "value"])

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Keep only rows where the watch was actually worn and signal is valid
        df = df[df["missing_value_reason"].isna()].copy()
        # The value column is whichever is not a known metadata column
        meta = {"timestamp_unix", "timestamp_iso",
                "participant_full_id", "missing_value_reason"}
        val_cols = [c for c in df.columns if c not in meta]
        if not val_cols:
            continue
        df = df[["timestamp_unix", val_cols[0]]].rename(columns={val_cols[0]: "value"})
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["timestamp_unix", "value"])

    out = pd.concat(dfs, ignore_index=True)
    out["timestamp_unix"] = out["timestamp_unix"].astype(np.int64)
    out["value"]          = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).sort_values("timestamp_unix").reset_index(drop=True)

    print(f"  {signal_suffix:15s}: {len(out):4d} valid rows  "
          f"({pd.to_datetime(out['timestamp_unix'].iloc[0],  unit='ms', utc=True).strftime('%m-%d %H:%M')} "
          f"-> "
          f"{pd.to_datetime(out['timestamp_unix'].iloc[-1], unit='ms', utc=True).strftime('%m-%d %H:%M')} UTC)")
    return out


def window_stats(sig_df: pd.DataFrame,
                 t_start_ms: int, t_end_ms: int) -> tuple[float, float]:
    """
    Slice sig_df to [t_start_ms - WINDOW_PAD_MS, t_end_ms] and return
    (mean, max).  Returns (nan, nan) when the window is empty.
    """
    sub = sig_df[
        (sig_df["timestamp_unix"] >= t_start_ms - WINDOW_PAD_MS) &
        (sig_df["timestamp_unix"] <= t_end_ms)
    ]["value"]
    if len(sub) == 0:
        return float("nan"), float("nan")
    return float(sub.mean()), float(sub.max())


def extract_features(t_start_ms: int, t_end_ms: int,
                     hr_df: pd.DataFrame,
                     prv_df: pd.DataFrame,
                     eda_df: pd.DataFrame) -> dict:
    """
    Return the 4 physiological features for the given clip window.
    Uses the single continuous Empatica stream — no per-user lookup needed.
    """
    hr_mean,  _        = window_stats(hr_df,  t_start_ms, t_end_ms)
    hrv_rmssd, _       = window_stats(prv_df, t_start_ms, t_end_ms)
    eda_mean,  eda_max = window_stats(eda_df, t_start_ms, t_end_ms)

    def r2(x): return round(x, 2) if not math.isnan(x) else float("nan")

    return {
        "hr_mean":   r2(hr_mean),
        "hrv_rmssd": r2(hrv_rmssd),
        "eda_mean":  round(eda_mean, 4) if not math.isnan(eda_mean) else float("nan"),
        "eda_max":   round(eda_max,  4) if not math.isnan(eda_max)  else float("nan"),
    }


# ─── LLM  ──────────────────────────────────────────────────────────────────

def fmt_feat(val: float, unit: str) -> str:
    if math.isnan(val):
        return "unavailable (watch not worn or no data in this window)"
    return f"{val:.2f} {unit}"


def build_prompt(row: pd.Series, feats: dict) -> str:
    clip      = str(row.get("clip_title", "unknown")).replace(".mp4", "")
    dur_s     = (float(row["timestamp_clip_end"]) -
                 float(row["timestamp_clip_start"])) / 1000.0
    p_val, p_aro   = row.get("p_val",  "?"), row.get("p_aro",  "?")
    i1_val, i1_aro = row.get("i1_val", "?"), row.get("i1_aro", "?")

    return f"""### Instruction
You are an intelligent healthcare agent specialising in affective computing and wearable physiological signal analysis.

### Health Knowledge
Valence (emotional positivity): 1 = very negative, 5 = very positive.
Arousal (activation level):     1 = very calm,     5 = very excited.
Clinical anchors:
  HR > 90 BPM  + RMSSD < 20 ms  =>  High Arousal, possibly Negative Valence (stress / fear / excitement)
  HR 65-80 BPM + RMSSD > 50 ms  =>  Low Arousal,  Positive Valence (calm / content)
  EDA > 3 uS  =>  sympathetic activation (high arousal / emotional stress)
  EDA < 0.5 uS => parasympathetic dominance (low arousal)

### Activity
The participant watched the video clip "{clip}" ({dur_s:.0f} seconds).

### Self-Reports  [context only -- do NOT copy these as your answer]
  Perceived  emotion (P) : Valence={p_val}, Arousal={p_aro}
  Induced    emotion (I1): Valence={i1_val}, Arousal={i1_aro}

### Physiological Sensor Readings  (Empatica EmbracePlus, 1-minute aggregated)
  Mean Heart Rate   (hr_mean)  : {fmt_feat(feats['hr_mean'],   'BPM')}
  HRV RMSSD         (hrv_rmssd): {fmt_feat(feats['hrv_rmssd'], 'ms')}
  Mean EDA          (eda_mean) : {fmt_feat(feats['eda_mean'],  'uS')}
  Peak EDA          (eda_max)  : {fmt_feat(feats['eda_max'],   'uS')}

### Question
Based solely on the physiological signals above, predict the PHYSIOLOGICALLY INDUCED emotion (I2) the participant experienced during this clip.  Use the integer 1-5 scale.

### Response
Return ONLY this JSON with no extra text:
{{"I2_val": <integer 1-5>, "I2_aro": <integer 1-5>}}""".strip()


_groq_client = None

def call_llm(prompt: str) -> dict | None:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
            )
            raw = resp.choices[0].message.content.strip().strip("`").replace("json", "").strip()
            parsed = json.loads(raw)
            i2v = int(round(float(parsed.get("I2_val", parsed.get("valence", 3)))))
            i2a = int(round(float(parsed.get("I2_aro", parsed.get("arousal", 3)))))
            return {"I2_val": max(1, min(5, i2v)), "I2_aro": max(1, min(5, i2a))}
        except json.JSONDecodeError as e:
            print(f"    [retry {attempt}] JSON error: {e}  raw={raw!r}")
        except Exception as e:
            print(f"    [retry {attempt}] error: {e}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


# ─── EXPRESSED EMOTION  ────────────────────────────────────────────────────

def load_expressed(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {str(r["clip_id"]): (float(r["expressed_valence"]), float(r["expressed_arousal"]))
            for _, r in df.iterrows()}


def get_equad(clip_title: str, e_dict: dict) -> str:
    cid = str(clip_title).replace(".mp4", "").strip()
    e = e_dict.get(cid)
    if e is None:
        return "Q0"
    v, a = e
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"


# ─── MAIN  ─────────────────────────────────────────────────────────────────

def main():
    SEP = "=" * 65
    print(SEP)
    print("  TASK-H: Physiological Extraction + LLM Inference (I2)")
    print(SEP)

    # 1. Load continuous Empatica stream (all date-split files merged)
    print("\n  Loading Empatica continuous stream ...")
    hr_df  = load_signal("pulse-rate")
    prv_df = load_signal("prv")
    eda_df = load_signal("eda")

    phys_start = hr_df["timestamp_unix"].min() if len(hr_df) else 0
    phys_end   = hr_df["timestamp_unix"].max() if len(hr_df) else 0

    # 2. Load clip responses
    df = pd.read_csv(RESPONSES_CSV, sep=";")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    print(f"\n  Loaded {len(df)} response rows, "
          f"{df['user_id'].nunique()} users")

    # 3. Filter to real stimulus blocks, then keep only rows whose clip
    #    window overlaps with the Empatica recording window
    df = df[df["block_type"].isin(VALID_BLOCKS)].copy()

    df["in_phys_window"] = (
        (df["timestamp_clip_end"]   >= phys_start - WINDOW_PAD_MS) &
        (df["timestamp_clip_start"] <= phys_end)
    )
    df_phys  = df[df["in_phys_window"]].copy()
    df_nophys = df[~df["in_phys_window"]].copy()

    print(f"\n  Stimulus rows inside  Empatica window: {len(df_phys)}")
    print(f"  Stimulus rows outside Empatica window: {len(df_nophys)} "
          f"(I2 will be NaN -- physio not recorded for those sessions)")

    # 4. Attach E_quadrant
    e_dict = load_expressed(EXPRESSED_CSV)
    df["E_quadrant"] = df["clip_title"].apply(lambda t: get_equad(t, e_dict))

    # 5. Init I2 columns
    for col in ("I2_val", "I2_aro"):
        if col not in df.columns:
            df[col] = float("nan")

    # 6. Infer I2 for each row inside the Empatica window
    total = len(df_phys)
    done  = 0

    print(f"\n  Running LLM on {total} rows ...\n")

    for idx in df_phys.index:
        row     = df.loc[idx]
        t_start = int(row["timestamp_clip_start"])
        t_end   = int(row["timestamp_clip_end"])
        uid     = int(row["user_id"])
        clip    = str(row["clip_title"])
        done   += 1

        tag = f"[{done:3d}/{total}]"

        # Skip already populated (idempotency for reruns)
        if not pd.isna(row.get("I2_val")) and not pd.isna(row.get("I2_aro")):
            print(f"  {tag} user={uid:2d} {clip:<35s} -> already populated, skip")
            continue

        feats = extract_features(t_start, t_end, hr_df, prv_df, eda_df)
        all_nan = all(math.isnan(v) for v in feats.values())

        if all_nan:
            print(f"  {tag} user={uid:2d} {clip:<35s} -> no data in window, I2=NaN")
            continue

        print(f"  {tag} user={uid:2d} {clip}")
        print(f"         hr={feats['hr_mean']} bpm | rmssd={feats['hrv_rmssd']} ms "
              f"| eda={feats['eda_mean']}/{feats['eda_max']} uS")

        result = call_llm(build_prompt(row, feats))
        if result:
            df.at[idx, "I2_val"] = result["I2_val"]
            df.at[idx, "I2_aro"] = result["I2_aro"]
            print(f"         -> I2_val={result['I2_val']}, I2_aro={result['I2_aro']}")
        else:
            print(f"         -> LLM failed after {MAX_RETRIES} retries, I2=NaN")

    # 7. Save full filtered set (includes rows without phys data, I2=NaN)
    df.to_csv(OUTPUT_CSV, index=False)

    i2_filled = int(df["I2_val"].notna().sum())
    print(f"\n{SEP}")
    print(f"  Done.  Rows saved: {len(df)}")
    print(f"  I2 populated   : {i2_filled}")
    print(f"  I2 NaN (no phys data): {len(df) - i2_filled}")
    print(f"  Output -> {OUTPUT_CSV}")
    print(SEP)


if __name__ == "__main__":
    main()
