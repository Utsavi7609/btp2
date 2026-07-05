"""
run_groq_llama.py
=================
INNER DIRECTORY: InHouseCollectedData/

Runs LLM inference using Llama-3.3-70B via Groq API.
Uses zero-shot prompting strategy from Health-LLM (Kim et al., 2024).

Llama 3.3 70B is used (stronger than 8B for reasoning tasks).
Health-LLM Table 3 shows Llama 2 benefits significantly from
larger parameter count and few-shot; we use 70B zero-shot here.

Input:  fitbit_ready_for_llm.csv
Output: llm_results_llama.csv
"""

import pandas as pd
from groq import Groq
import time
import json
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
MODEL        = "llama-3.3-70b-versatile"   # stronger than llama-3.1-8b-instant
INPUT_FILE   = "fitbit_ready_for_llm.csv"
OUTPUT_FILE  = "llm_results_llama.csv"

# Temperature 0 → deterministic outputs for reproducibility
# Health-LLM uses greedy decoding (do_sample=False) for evaluation
TEMPERATURE  = 0.0
MAX_TOKENS   = 60   # {"valence": X, "arousal": Y} needs very few tokens

# ─── SETUP ────────────────────────────────────────────────────────────────────
client   = Groq(api_key=GROQ_API_KEY)
df_all   = pd.read_csv(INPUT_FILE)

# Resume logic: read how many rows already done
if os.path.exists(OUTPUT_FILE):
    df_done   = pd.read_csv(OUTPUT_FILE)
    start_row = len(df_done)
    print(f"Resuming from row {start_row} / {len(df_all)}")
else:
    start_row = 0
    # Write header row
    header_cols = list(df_all.columns) + ['inferred_valence', 'inferred_arousal', 'raw_response']
    pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE, index=False)
    print(f"Starting fresh. {len(df_all)} rows to process.")

print(f"Model: {MODEL}\n")

# ─── INFERENCE LOOP ───────────────────────────────────────────────────────────
for i in range(start_row, len(df_all)):
    row = df_all.iloc[i]
    print(f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
          end="  ", flush=True)

    inf_valence  = None
    inf_arousal  = None
    raw_response = ""
    retries      = 0

    while retries < 5:
        try:
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": row['llm_question']}],
                model=MODEL,
                response_format={"type": "json_object"},  # guarantees valid JSON
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_response = completion.choices[0].message.content.strip()
            res          = json.loads(raw_response)

            inf_valence  = res.get('valence')
            inf_arousal  = res.get('arousal')

            print(f"V={inf_valence}  A={inf_arousal}")
            time.sleep(1.2)   # stay within Groq rate limits
            break

        except Exception as e:
            retries += 1
            err_str = str(e)

            if "429" in err_str or "rate" in err_str.lower():
                wait = 45 * retries
                print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
                time.sleep(wait)
            elif "json" in err_str.lower():
                # Bad JSON response; try to extract numbers
                print(f"\n  [JSON parse error] Raw: {raw_response}")
                print(f"  Retrying... (attempt {retries}/5)")
                time.sleep(10)
            else:
                print(f"\n  [Error] {err_str}")
                time.sleep(15)

            if retries >= 5:
                print(f"  [Skip] Failed 5 times. Recording null.")
                raw_response = f"ERROR after 5 retries: {err_str}"

    # Write this row immediately (no data loss on crash)
    new_row = list(row) + [inf_valence, inf_arousal, raw_response]
    pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")

# Quick sanity check
df_out = pd.read_csv(OUTPUT_FILE)
df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')
valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
failures = len(df_out) - len(valid)
print(f"   Valid predictions: {len(valid)} / {len(df_out)}")
print(f"   Failures / nulls:  {failures}")
