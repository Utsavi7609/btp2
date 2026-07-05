# import os
# import pandas as pd
# import numpy as np
# import time
# import json
# from groq import Groq

# # ───────── CONFIG ─────────
# GROQ_API_KEY = "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"
# MODEL = "qwen/qwen3-32b"

# BASE = "../K-EmoCon/extracted"

# client = Groq(api_key=GROQ_API_KEY)

# # ───────── STEP 1: BUILD DATASET ─────────

# rows = []

# participants = os.listdir(f"{BASE}/e4_data")

# for p in participants:

#     hr_path = f"{BASE}/e4_data/{p}/HR.csv"
#     ann_path = f"{BASE}/emotion_annotations/self_annotations/{p}.csv"

#     if not os.path.exists(hr_path) or not os.path.exists(ann_path):
#         continue

#     # load HR
#     with open(hr_path, "r") as f:
#         lines = f.readlines()

#     start = float(lines[0].strip())
#     sr = float(lines[1].strip())

#     hr_vals = [float(x.strip()) for x in lines[2:]]

#     times = [start + i*(1/sr) for i in range(len(hr_vals))]

#     hr_df = pd.DataFrame({"time": times, "hr": hr_vals})

#     # load annotations
#     ann_df = pd.read_csv(ann_path)

#     for _, r in ann_df.iterrows():

#         t = r["timestamp"]

#         window = hr_df[(hr_df["time"] >= t-60) & (hr_df["time"] <= t)]

#         if len(window) < 10:
#             continue

#         seq = window["hr"].tolist()

#         avg = np.mean(seq)
#         std = np.std(seq)
#         mn = np.min(seq)
#         mx = np.max(seq)
#         drift = seq[-1] - seq[0]

#         diffs = np.diff(seq)
#         rmssd = np.sqrt(np.mean(diffs**2)) if len(diffs)>0 else 0

#         rows.append({
#             "participant": p,
#             "clip_title": "kemocon",
#             "clip_order": 0,
#             "timestamp": t,
#             "hr_sequence": seq,
#             "avg_bpm": avg,
#             "hr_std": std,
#             "hr_min": mn,
#             "hr_max": mx,
#             "bpm_drift": drift,
#             "hr_count": len(seq),
#             "hrv_rmssd": rmssd,
#             "self_reported_valence": r["valence"],
#             "self_reported_arousal": r["arousal"]
#         })

# df = pd.DataFrame(rows)

# print("Rows created:", len(df))

# # ───────── STEP 2: BUILD PROMPTS ─────────

# def build_prompt(row):

#     return f"""
# You are an expert in affective computing.

# Valence: 1 (very negative) → 5 (very positive)
# Arousal: 1 (calm) → 5 (excited)

# Heart rate increases with arousal.
# HR variability reflects emotional intensity.

# Data:
# Mean={row['avg_bpm']}, Std={row['hr_std']}, Min={row['hr_min']}, Max={row['hr_max']}, Drift={row['bpm_drift']}, RMSSD={row['hrv_rmssd']}

# Predict valence and arousal.

# Return ONLY JSON:
# {{"valence": X, "arousal": Y}}
# """

# df["prompt"] = df.apply(build_prompt, axis=1)

# # ───────── STEP 3: RUN LLM ─────────

# results = []

# for i, row in df.iterrows():

#     try:

#         completion = client.chat.completions.create(
#             model=MODEL,
#             messages=[{"role":"user","content":row["prompt"]}],
#             temperature=0,
#             max_tokens=100
#         )

#         text = completion.choices[0].message.content

#         # extract json
#         try:
#             parsed = json.loads(text)
#         except:
#             import re
#             m = re.search(r"\{.*?\}", text)
#             parsed = json.loads(m.group()) if m else {}

#         v = parsed.get("valence")
#         a = parsed.get("arousal")

#     except Exception as e:
#         print("Error:", e)
#         v, a = None, None

#     results.append((v,a))

#     print(f"{i+1}/{len(df)} → V={v} A={a}")

#     time.sleep(2)

# df["pred_valence"] = [x[0] for x in results]
# df["pred_arousal"] = [x[1] for x in results]

# # ───────── STEP 4: EVALUATE ─────────

# from sklearn.metrics import mean_absolute_error

# df_clean = df.dropna()

# v_true = df_clean["self_reported_valence"]
# a_true = df_clean["self_reported_arousal"]

# v_pred = df_clean["pred_valence"]
# a_pred = df_clean["pred_arousal"]

# mae_v = mean_absolute_error(v_true, v_pred)
# mae_a = mean_absolute_error(a_true, a_pred)

# corr_v = np.corrcoef(v_true, v_pred)[0,1]
# corr_a = np.corrcoef(a_true, a_pred)[0,1]

# acc_v = np.mean(np.abs(v_true - v_pred) <= 1)
# acc_a = np.mean(np.abs(a_true - a_pred) <= 1)

# print("\n===== FINAL RESULTS =====")
# print("MAE Valence:", mae_v)
# print("MAE Arousal:", mae_a)
# print("Corr Valence:", corr_v)
# print("Corr Arousal:", corr_a)
# print("Acc Valence:", acc_v)
# print("Acc Arousal:", acc_a)


import pandas as pd
import os
from groq import Groq
import json
import time
import random

# ================= CONFIG =================
GROQ_API_KEY = "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"
MODEL = "qwen/qwen3-32b"

BASE = "../K-EmoCon/extracted"
E4_PATH = f"{BASE}/e4_data"
ANN_PATH = f"{BASE}/emotion_annotations/self_annotations"

OUTPUT_FILE = "kemocon_llm_results.csv"

client = Groq(api_key=GROQ_API_KEY)

# ================= STEP 1: BUILD DATASET =================

rows = []

participants = os.listdir(E4_PATH)
print("Participants found:", participants[:5])

for p in participants:

    e4_folder = f"{E4_PATH}/{p}"
    hr_file = f"{e4_folder}/E4_HR.csv"
    ann_file = f"{ANN_PATH}/P{p}.self.csv"

    # skip if files missing
    if not os.path.exists(hr_file) or not os.path.exists(ann_file):
        continue

    try:
        hr_df = pd.read_csv(hr_file, header=None)
        ann_df = pd.read_csv(ann_file)
    except:
        continue

    # HR values (skip first row metadata if exists)
    hr_values = hr_df.iloc[1:, 0].astype(float).values

    if len(hr_values) < 10:
        continue

    # basic stats
    avg_bpm = hr_values.mean()
    std_bpm = hr_values.std()
    min_bpm = hr_values.min()
    max_bpm = hr_values.max()
    drift = hr_values[-1] - hr_values[0]

    # annotation columns (IMPORTANT: adjust if needed)
    if "valence" not in ann_df.columns or "arousal" not in ann_df.columns:
        continue

    valence = ann_df["valence"].mean()
    arousal = ann_df["arousal"].mean()

    rows.append({
        "participant": p,
        "avg_bpm": avg_bpm,
        "std_bpm": std_bpm,
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "drift": drift,
        "true_valence": valence,
        "true_arousal": arousal
    })

df = pd.DataFrame(rows)

print(f"Rows created: {len(df)}")

if len(df) == 0:
    print("❌ No data found. Check paths/columns.")
    exit()

# ================= STEP 2: BUILD PROMPT =================

def build_prompt(row):
    return f"""
You are an expert in affective computing.

Heart Rate Stats:
- Mean BPM: {row['avg_bpm']:.2f}
- Std BPM: {row['std_bpm']:.2f}
- Min BPM: {row['min_bpm']:.2f}
- Max BPM: {row['max_bpm']:.2f}
- Drift: {row['drift']:.2f}

Predict emotional state:
Valence (1–5), Arousal (1–5)

Return ONLY JSON:
{{"valence": X, "arousal": Y}}
"""

df["prompt"] = df.apply(build_prompt, axis=1)

# ================= STEP 3: LLM INFERENCE =================

pred_val = []
pred_aro = []
raw_out = []

for i, row in df.iterrows():

    print(f"[{i+1}/{len(df)}]", end=" ")

    retries = 0
    val, aro = None, None
    response_text = ""

    while retries < 5:
        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": row["prompt"]}],
                model=MODEL,
                temperature=0.0,
                max_tokens=50
            )

            response_text = completion.choices[0].message.content.strip()

            # try parse JSON
            try:
                parsed = json.loads(response_text)
            except:
                import re
                match = re.search(r"\{.*?\}", response_text)
                if match:
                    parsed = json.loads(match.group())
                else:
                    parsed = {}

            val = parsed.get("valence")
            aro = parsed.get("arousal")

            print(f"V={val} A={aro}")

            time.sleep(2 + random.uniform(0.5,1.0))
            break

        except Exception as e:
            retries += 1
            print(f"\nRetry {retries}: {e}")
            time.sleep(20)

    pred_val.append(val)
    pred_aro.append(aro)
    raw_out.append(response_text)

# ================= STEP 4: SAVE =================

df["pred_valence"] = pred_val
df["pred_arousal"] = pred_aro
df["raw_response"] = raw_out

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Saved to {OUTPUT_FILE}")