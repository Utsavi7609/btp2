import pandas as pd
from groq import Groq
import time
import json
import os
import random
import re

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
MODEL        = "llama-3.3-70b-versatile"
RESULTS_FILE = "llm_results_llama_fewshot.csv"
INPUT_FILE   = "fitbit_ready_for_llm.csv"

client = Groq(api_key=GROQ_API_KEY)
df_all = pd.read_csv(INPUT_FILE)

if not os.path.exists(RESULTS_FILE):
    print(f"Error: {RESULTS_FILE} not found.")
    exit(1)

df_results = pd.read_csv(RESULTS_FILE)

# Identify nulls
null_mask = df_results['inferred_valence'].isna() | df_results['inferred_arousal'].isna()
null_indices = df_results[null_mask].index.tolist()

if not null_indices:
    print("✅ No null values found in results. Dataset is complete!")
    exit(0)

print(f"Found {len(null_indices)} rows with null values. Starting recovery...")

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_sensor_text(prompt_str):
    match = re.search(r'(### Sensor Readings: .*?)\n\n### Question:', prompt_str, re.DOTALL)
    if match:
        return match.group(1).replace('### Sensor Readings: ', '').strip()
    return None

def build_few_shot_prompt(row_idx, row_data, df):
    base_prompt = str(row_data['llm_question'])
    valid_candidates = df[
        (df['participant'] != row_data['participant']) & 
        (df['participant_valence'].notna()) & 
        (df['participant_arousal'].notna())
    ]
    few_shot_examples = "### Here are a few reference examples mapping physiological data to true scores:\n\n"
    if len(valid_candidates) >= 3:
        sampled = valid_candidates.sample(3, random_state=row_idx)
        for _, ex_row in sampled.iterrows():
            ex_sensors = extract_sensor_text(str(ex_row['llm_question']))
            if ex_sensors:
                ex_v = ex_row['participant_valence']
                ex_a = ex_row['participant_arousal']
                few_shot_examples += (
                    f"**Example Input:**\n"
                    f"Sensor Readings: {ex_sensors}\n"
                    f"**Example Expected Output JSON:**\n"
                    f"{{\"valence\": {ex_v}, \"arousal\": {ex_a}}}\n\n"
                )
    
    injected_prompt = base_prompt.replace(
        "### Sensor Readings: ",
        few_shot_examples + "### CURRENT TARGET PARTICIPANT INFERENCE:\n### Sensor Readings: "
    )
    return injected_prompt

# ─── RECOVERY LOOP ────────────────────────────────────────────────────────────
for idx in null_indices:
    row = df_all.iloc[idx]
    print(f"Recovering row {idx}: {row['participant']} | {row['clip_title'][:30]}...", end=" ", flush=True)

    inf_valence = None
    inf_arousal = None
    raw_response = ""
    retries = 0

    while retries < 5:
        try:
            # Pacing
            time.sleep(20 + random.uniform(1, 5))
            
            prompt_text = build_few_shot_prompt(idx, row, df_all)
            completion = client.chat.completions.create(
                messages=[{"role":"user","content":prompt_text}],
                model=MODEL,
                response_format={"type":"json_object"},
                temperature=0.0,
                max_tokens=60
            )

            raw_response = completion.choices[0].message.content.strip()
            res = json.loads(raw_response)
            inf_valence = res.get("valence")
            inf_arousal = res.get("arousal")

            if inf_valence is not None and inf_arousal is not None:
                # Update the dataframe
                df_results.at[idx, 'inferred_valence'] = inf_valence
                df_results.at[idx, 'inferred_arousal'] = inf_arousal
                df_results.at[idx, 'raw_response'] = raw_response
                print(f"Success! V={inf_valence} A={inf_arousal}")
                # Save after each success to be safe
                df_results.to_csv(RESULTS_FILE, index=False)
                break
            else:
                print("?", end="", flush=True)
                retries += 1

        except Exception as e:
            retries += 1
            err_str = str(e).lower()
            if "429" in err_str or "rate" in err_str:
                wait_match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait = float(wait_match.group(1)) + 5 if wait_match else 60
                print(f"(wait {int(wait)}s)", end="", flush=True)
                time.sleep(wait)
            elif "401" in err_str or "quota" in err_str:
                print("\n[Quota Reached] Please update the API key in the script.")
                exit(1)
            else:
                print("E", end="", flush=True)
                time.sleep(5)

print("\nDone! Recovery complete.")
