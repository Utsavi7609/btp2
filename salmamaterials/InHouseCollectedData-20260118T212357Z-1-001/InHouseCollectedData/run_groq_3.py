
# import pandas as pd
# from groq import Groq
# import time
# import json
# import os
# import random
# import re

# # ─── CONFIGURATION ────────────────────────────────────────────────────────────
# GROQ_API_KEY = "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"
# # MODEL = "meta/llama-4-scout-17b-16e-instruct"
# MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# INPUT_FILE   = "fitbit_ready_for_llm.csv"
# OUTPUT_FILE = "llm_results_llama4.csv"

# # OUTPUT_FILE  = "llm_results_mistral.csv"

# TEMPERATURE  = 0.0
# MAX_TOKENS   = 160

# # ─── SETUP ────────────────────────────────────────────────────────────────────
# client = Groq(api_key=GROQ_API_KEY)
# df_all = pd.read_csv(INPUT_FILE)

# # Auto-clean if previous run produced all nulls
# if os.path.exists(OUTPUT_FILE):
#     df_check = pd.read_csv(OUTPUT_FILE)
#     df_check['inferred_valence'] = pd.to_numeric(df_check['inferred_valence'], errors='coerce')
#     if df_check['inferred_valence'].isna().all() and len(df_check) > 0:
#         print("Previous output file has all nulls — deleting and starting fresh.")
#         os.remove(OUTPUT_FILE)

# if os.path.exists(OUTPUT_FILE):
#     df_done   = pd.read_csv(OUTPUT_FILE)
#     start_row = len(df_done)
#     print(f"Resuming from row {start_row} / {len(df_all)}")
# else:
#     start_row = 0
#     header_cols = list(df_all.columns) + ['inferred_valence', 'inferred_arousal', 'raw_response']
#     pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE, index=False)
#     print(f"Starting fresh. {len(df_all)} rows to process.")

# print(f"Model: {MODEL}")
# print("Thinking mode disabled via /no_think directive\n")


# # ─── HELPERS ──────────────────────────────────────────────────────────────────

# def strip_thinking(text):
#     """Remove <think>...</think> blocks that qwen3 sometimes emits."""
#     return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# def extract_json_robust(text):
#     """
#     Extract valence/arousal from model output.
#     Three fallback layers so we never miss a valid response.
#     """
#     text = strip_thinking(text)
#     if not text:
#         return None

#     # Layer 1: whole text is valid JSON
#     try:
#         obj = json.loads(text)
#         if 'valence' in obj and 'arousal' in obj:
#             return obj
#     except Exception:
#         pass

#     # Layer 2: find first { ... } block
#     match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
#     if match:
#         try:
#             obj = json.loads(match.group())
#             if 'valence' in obj and 'arousal' in obj:
#                 return obj
#         except Exception:
#             pass

#     # Layer 3: regex extract numbers
#     v = re.search(r'["\s]?[Vv]alence["\s]*[:=]\s*([1-5])', text)
#     a = re.search(r'["\s]?[Aa]rousal["\s]*[:=]\s*([1-5])', text)
#     if v and a:
#         return {'valence': int(v.group(1)), 'arousal': int(a.group(1))}

#     return None


# def build_prompt(base_prompt):
#     """
#     Append /no_think and tighten the output instruction.
#     qwen3 docs: /no_think in the user turn suppresses the <think> block.
#     """
#     # Replace the response instruction line to be maximally explicit
#     prompt = re.sub(
#         r'### Response:.*',
#         '### Response: Output ONLY valid JSON, nothing else: {"valence": X, "arousal": Y}\n\n/no_think',
#         base_prompt,
#         flags=re.DOTALL
#     )
#     return prompt


# # ─── INFERENCE LOOP ───────────────────────────────────────────────────────────
# for i in range(start_row, len(df_all)):
#     row = df_all.iloc[i]
#     print(
#         f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
#         end="  ", flush=True
#     )

#     prompt_text  = build_prompt(str(row['llm_question']))
#     inf_valence  = None
#     inf_arousal  = None
#     raw_response = ""
#     retries      = 0

#     while retries < 5:
#         try:
#             # No response_format parameter — qwen3 rejects json_object mode
#             completion = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": prompt_text}],
#                 temperature=TEMPERATURE,
#                 max_tokens=MAX_TOKENS,
#             )

#             raw_response = completion.choices[0].message.content.strip()
#             parsed       = extract_json_robust(raw_response)

#             if parsed:
#                 inf_valence = parsed.get('valence')
#                 inf_arousal = parsed.get('arousal')
#                 print(f"V={inf_valence}  A={inf_arousal}")
#             else:
#                 cleaned = strip_thinking(raw_response)
#                 print(f"V=None  A=None  [got: {repr(cleaned[:80])}]")

#             time.sleep(1.5 + random.uniform(0, 0.4))
#             break

#         except Exception as e:
#             retries += 1
#             err_str = str(e)

#             if "429" in err_str or "rate" in err_str.lower():
#                 wait = 60 * retries
#                 print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
#                 time.sleep(wait)

#             elif "decommissioned" in err_str or "model_not_found" in err_str:
#                 print(f"\n  [Model error] {MODEL} not available.")
#                 print("  Available models on your account:")
#                 print("    qwen/qwen3-32b, moonshotai/kimi-k2-instruct,")
#                 print("    openai/gpt-oss-20b, openai/gpt-oss-120b,")
#                 print("    meta-llama/llama-4-scout-17b-16e-instruct")
#                 import sys; sys.exit(1)

#             else:
#                 print(f"\n  [Error] {err_str[:120]}")
#                 time.sleep(15)

#             if retries >= 5:
#                 print("  [Skip] Failed 5 times. Recording null.")
#                 raw_response = f"FAILED: {err_str[:100]}"

#     new_row = list(row) + [inf_valence, inf_arousal, raw_response]
#     pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# # ─── SUMMARY ──────────────────────────────────────────────────────────────────
# print(f"\nDone. Results saved to {OUTPUT_FILE}")

# df_out = pd.read_csv(OUTPUT_FILE)
# df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
# df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')

# valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
# failures = len(df_out) - len(valid)

# print(f"Valid predictions: {len(valid)} / {len(df_out)}")
# print(f"Failures / nulls:  {failures}")

# # Debug: show what failed rows actually returned
# if failures > 0:
#     print(f"\nSample of failed raw responses (to diagnose any remaining issues):")
#     sample = df_out[df_out['inferred_valence'].isna()]['raw_response'].head(3)
#     for idx, raw in sample.items():
#         print(f"  Row {idx}: {repr(strip_thinking(str(raw))[:150])}")


"""
run_groq_3.py
=============
INNER DIRECTORY: InHouseCollectedData/

Llama-4-Scout inference. Uses response_format=json_object (supported by all
Llama models on Groq) instead of /no_think (qwen3-only directive).
"""

import pandas as pd
from groq import Groq
import time
import json
import os
import random
import re

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
MODEL        = "meta-llama/llama-4-scout-17b-16e-instruct"
INPUT_FILE   = "fitbit_ready_for_llm.csv"
OUTPUT_FILE  = "llm_results_llama4_fewshot.csv"

TEMPERATURE  = 0.0
MAX_TOKENS   = 30   # {"valence": X, "arousal": Y} is only ~12 tokens

# ─── SETUP ────────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)
df_all = pd.read_csv(INPUT_FILE)

# Auto-clean if previous run produced all nulls
if os.path.exists(OUTPUT_FILE):
    df_check = pd.read_csv(OUTPUT_FILE)
    df_check['inferred_valence'] = pd.to_numeric(df_check['inferred_valence'], errors='coerce')
    if df_check['inferred_valence'].isna().all() and len(df_check) > 0:
        print("Previous output has all nulls — deleting and starting fresh.")
        os.remove(OUTPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    df_done   = pd.read_csv(OUTPUT_FILE)
    start_row = len(df_done)
    print(f"Resuming from row {start_row} / {len(df_all)}")
else:
    start_row = 0
    header_cols = list(df_all.columns) + ['inferred_valence', 'inferred_arousal', 'raw_response']
    pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE, index=False)
    print(f"Starting fresh. {len(df_all)} rows to process.")

print(f"Model: {MODEL}\n")

# ─── FEW-SHOT INJECTION LOGIC ─────────────────────────────────────────────────
def extract_sensor_text(prompt_str):
    """
    Reverted to full extraction for consistency as requested.
    """
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

# ─── INFERENCE LOOP ───────────────────────────────────────────────────────────
for i in range(start_row, len(df_all)):
    row = df_all.iloc[i]
    print(
        f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
        end="  ", flush=True
    )

    inf_valence  = None
    inf_arousal  = None
    raw_response = ""
    retries      = 0

    while retries < 5:
        try:
            prompt_text = build_few_shot_prompt(i, row, df_all)
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},  # forces clean JSON, no reasoning steps
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw_response = completion.choices[0].message.content.strip()
            res          = json.loads(raw_response)
            inf_valence  = res.get('valence')
            inf_arousal  = res.get('arousal')

            print(f"V={inf_valence}  A={inf_arousal}")
            time.sleep(1.5 + random.uniform(0, 0.4))
            break

        except Exception as e:
            retries += 1
            err_str = str(e)

            if "429" in err_str or "rate" in err_str.lower():
                wait = 60 * retries
                print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
                time.sleep(wait)

            elif "decommissioned" in err_str or "not_found" in err_str:
                print(f"\n  [Model error] {MODEL} not available. Check model name.")
                import sys; sys.exit(1)

            else:
                print(f"\n  [Error] {err_str[:120]}")
                time.sleep(15)

            if retries >= 5:
                print("  [Skip] Failed 5 times. Recording null.")
                raw_response = f"FAILED: {err_str[:100]}"

    new_row = list(row) + [inf_valence, inf_arousal, raw_response]
    pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nDone. Results saved to {OUTPUT_FILE}")

df_out = pd.read_csv(OUTPUT_FILE)
df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')

valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
failures = len(df_out) - len(valid)

print(f"Valid predictions: {len(valid)} / {len(df_out)}")
print(f"Failures / nulls:  {failures}")

if failures > 0:
    print(f"\nSample of failed raw responses:")
    sample = df_out[df_out['inferred_valence'].isna()]['raw_response'].head(3)
    for idx, raw in sample.items():
        print(f"  Row {idx}: {repr(str(raw)[:150])}")