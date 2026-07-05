# # """
# # run_groq_llama.py
# # =================
# # INNER DIRECTORY: InHouseCollectedData/

# # Runs LLM inference using Llama-3.3-70B via Groq API.
# # Uses zero-shot prompting strategy from Health-LLM (Kim et al., 2024).

# # Llama 3.3 70B is used (stronger than 8B for reasoning tasks).
# # Health-LLM Table 3 shows Llama 2 benefits significantly from
# # larger parameter count and few-shot; we use 70B zero-shot here.

# # Input:  fitbit_ready_for_llm.csv
# # Output: llm_results_llama.csv
# # """

# # import pandas as pd
# # from groq import Groq
# # import time
# # import json
# # import os

# # # ─── CONFIGURATION ────────────────────────────────────────────────────────────
# # GROQ_API_KEY = "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"
# # MODEL        = "llama-3.3-70b-versatile"   # stronger than llama-3.1-8b-instant
# # INPUT_FILE   = "fitbit_ready_for_llm.csv"
# # OUTPUT_FILE  = "llm_results_llama.csv"

# # # Temperature 0 → deterministic outputs for reproducibility
# # # Health-LLM uses greedy decoding (do_sample=False) for evaluation
# # TEMPERATURE  = 0.0
# # MAX_TOKENS   = 60   # {"valence": X, "arousal": Y} needs very few tokens

# # # ─── SETUP ────────────────────────────────────────────────────────────────────
# # client   = Groq(api_key=GROQ_API_KEY)
# # df_all   = pd.read_csv(INPUT_FILE)

# # # Resume logic: read how many rows already done
# # if os.path.exists(OUTPUT_FILE):
# #     df_done   = pd.read_csv(OUTPUT_FILE)
# #     start_row = len(df_done)
# #     print(f"Resuming from row {start_row} / {len(df_all)}")
# # else:
# #     start_row = 0
# #     # Write header row
# #     header_cols = list(df_all.columns) + ['inferred_valence', 'inferred_arousal', 'raw_response']
# #     pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE, index=False)
# #     print(f"Starting fresh. {len(df_all)} rows to process.")

# # print(f"Model: {MODEL}\n")

# # # ─── INFERENCE LOOP ───────────────────────────────────────────────────────────
# # for i in range(start_row, len(df_all)):
# #     row = df_all.iloc[i]
# #     print(f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
# #           end="  ", flush=True)

# #     inf_valence  = None
# #     inf_arousal  = None
# #     raw_response = ""
# #     retries      = 0

# #     while retries < 5:
# #         try:
# #             completion = client.chat.completions.create(
# #                 messages=[{"role": "user", "content": row['llm_question']}],
# #                 model=MODEL,
# #                 response_format={"type": "json_object"},  # guarantees valid JSON
# #                 temperature=TEMPERATURE,
# #                 max_tokens=MAX_TOKENS,
# #             )
# #             raw_response = completion.choices[0].message.content.strip()
# #             res          = json.loads(raw_response)

# #             inf_valence  = res.get('valence')
# #             inf_arousal  = res.get('arousal')

# #             print(f"V={inf_valence}  A={inf_arousal}")
# #             time.sleep(1.2)   # stay within Groq rate limits
# #             break

# #         except Exception as e:
# #             retries += 1
# #             err_str = str(e)

# #             if "429" in err_str or "rate" in err_str.lower():
# #                 wait = 45 * retries
# #                 print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
# #                 time.sleep(wait)
# #             elif "json" in err_str.lower():
# #                 # Bad JSON response; try to extract numbers
# #                 print(f"\n  [JSON parse error] Raw: {raw_response}")
# #                 print(f"  Retrying... (attempt {retries}/5)")
# #                 time.sleep(10)
# #             else:
# #                 print(f"\n  [Error] {err_str}")
# #                 time.sleep(15)

# #             if retries >= 5:
# #                 print(f"  [Skip] Failed 5 times. Recording null.")
# #                 raw_response = f"ERROR after 5 retries: {err_str}"

# #     # Write this row immediately (no data loss on crash)
# #     new_row = list(row) + [inf_valence, inf_arousal, raw_response]
# #     pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# # print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")

# # # Quick sanity check
# # df_out = pd.read_csv(OUTPUT_FILE)
# # df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
# # df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')
# # valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
# # failures = len(df_out) - len(valid)
# # print(f"   Valid predictions: {len(valid)} / {len(df_out)}")
# # print(f"   Failures / nulls:  {failures}")



# import pandas as pd
# from groq import Groq
# import time
# import json
# import os
# import random

# # ─── CONFIGURATION ────────────────────────────────────────────────────────────
# GROQ_API_KEY = "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"
# MODEL        = "llama-3.3-70b-versatile"
# INPUT_FILE   = "fitbit_ready_for_llm.csv"
# OUTPUT_FILE  = "llm_results_llama.csv"

# TEMPERATURE  = 0.0
# MAX_TOKENS   = 30

# client   = Groq(api_key=GROQ_API_KEY)
# df_all   = pd.read_csv(INPUT_FILE)

# if os.path.exists(OUTPUT_FILE):
#     df_done   = pd.read_csv(OUTPUT_FILE)
#     start_row = len(df_done)
#     print(f"Resuming from row {start_row} / {len(df_all)}")
# else:
#     start_row = 0
#     header_cols = list(df_all.columns) + ['inferred_valence','inferred_arousal','raw_response']
#     pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE,index=False)
#     print(f"Starting fresh. {len(df_all)} rows to process.")

# print(f"Model: {MODEL}\n")

# for i in range(start_row,len(df_all)):

#     row=df_all.iloc[i]

#     print(f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",end="  ",flush=True)

#     inf_valence=None
#     inf_arousal=None
#     raw_response=""
#     retries=0

#     while retries<5:

#         try:

#             completion=client.chat.completions.create(
#                 messages=[{"role":"user","content":row['llm_question']}],
#                 model=MODEL,
#                 response_format={"type":"json_object"},
#                 temperature=TEMPERATURE,
#                 max_tokens=MAX_TOKENS,
#             )

#             raw_response=completion.choices[0].message.content.strip()
#             res=json.loads(raw_response)

#             inf_valence=res.get('valence')
#             inf_arousal=res.get('arousal')

#             print(f"V={inf_valence}  A={inf_arousal}")

#             # safer pacing for Groq rate limits
#             time.sleep(1.6 + random.uniform(0,0.4))

#             break

#         except Exception as e:

#             retries+=1
#             err_str=str(e)

#             if "429" in err_str or "rate" in err_str.lower():

#                 wait=60*retries
#                 print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
#                 time.sleep(wait)

#             elif "json" in err_str.lower():

#                 print(f"\n  [JSON parse error] Raw: {raw_response}")
#                 print(f"  Retrying... (attempt {retries}/5)")
#                 time.sleep(10)

#             else:

#                 print(f"\n  [Error] {err_str}")
#                 time.sleep(15)

#             if retries>=5:
#                 print(f"  [Skip] Failed 5 times. Recording null.")
#                 raw_response=f"ERROR after 5 retries: {err_str}"

#     new_row=list(row)+[inf_valence,inf_arousal,raw_response]
#     pd.DataFrame([new_row]).to_csv(OUTPUT_FILE,mode='a',header=False,index=False)

# print(f"\nDone. Results saved to {OUTPUT_FILE}")

# df_out=pd.read_csv(OUTPUT_FILE)
# df_out['inferred_valence']=pd.to_numeric(df_out['inferred_valence'],errors='coerce')
# df_out['inferred_arousal']=pd.to_numeric(df_out['inferred_arousal'],errors='coerce')
# valid=df_out.dropna(subset=['inferred_valence','inferred_arousal'])
# failures=len(df_out)-len(valid)

# print(f"Valid predictions: {len(valid)} / {len(df_out)}")
# print(f"Failures / nulls: {failures}")


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
INPUT_FILE   = "fitbit_ready_for_llm.csv"
OUTPUT_FILE  = "llm_results_llama_fewshot.csv"

TEMPERATURE  = 0.0
MAX_TOKENS   = 30

client   = Groq(api_key=GROQ_API_KEY)
df_all   = pd.read_csv(INPUT_FILE)

# ─── Resume logic ─────────────────────────────────────────────────────────────
if os.path.exists(OUTPUT_FILE):
    df_done   = pd.read_csv(OUTPUT_FILE)
    start_row = len(df_done)
    print(f"Resuming from row {start_row} / {len(df_all)}")
else:
    start_row = 0
    header_cols = list(df_all.columns) + ['inferred_valence','inferred_arousal','raw_response']
    pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE,index=False)
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
        end="  ",
        flush=True
    )

    # ─── AGGRESSIVE PACING FOR 70B ───
    # Few-shot prompts with full sequences are huge (~2-3k tokens).
    # Most Groq tiers have low TPM (Tokens Per Minute). 
    # 45s delay = 1.3 Requests Per Minute = ~4,000 TPM (Safe for 70B)
    time.sleep(10 + random.uniform(0.1, 5.0))

    inf_valence = None
    inf_arousal = None
    raw_response = ""
    retries = 0

    while retries < 5:

        try:
            prompt_text = build_few_shot_prompt(i, row, df_all)
            completion = client.chat.completions.create(
                messages=[{"role":"user","content":prompt_text}],
                model=MODEL,
                response_format={"type":"json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            raw_response = completion.choices[0].message.content.strip()
            res = json.loads(raw_response)

            inf_valence = res.get("valence")
            inf_arousal = res.get("arousal")

            print(f"V={inf_valence}  A={inf_arousal}")
            break

        except Exception as e:

            retries += 1
            err_str = str(e)

            if "429" in err_str or "rate" in err_str.lower():

                # Precision Waiting: Extract exact seconds if Groq provides them
                # Example: "Rate limit reached... Try again in 42.1s."
                wait_match = re.search(r"try again in ([\d\.]+)s", err_str.lower())
                if wait_match:
                    wait = float(wait_match.group(1)) + 2.5
                    print(f"\n  [Rate limit] Groq requested {wait_match.group(1)}s lock. Sleeping {wait:.1f}s...")
                else:
                    wait = 60.0 + (30.0 * retries)
                    print(f"\n  [Rate limit] Pacing down... Waiting {wait}s (attempt {retries}/5)...")
                
                time.sleep(wait)

            elif "json" in err_str.lower():

                print(f"\n  [JSON parse error] Raw: {raw_response}")
                print(f"  Retrying... (attempt {retries}/5)")
                time.sleep(10)

            else:
                err_str = str(e).lower()
                if "401" in err_str or "auth" in err_str or "api_key" in err_str or "quota" in err_str:
                    print(f"\n\n[FATAL ERROR] {e}")
                    new_key = input(">>> Current key failed/expired. Paste a NEW API Key (or press Ctrl+C to quit): ").strip()
                    if new_key:
                        GROQ_API_KEY = new_key
                        client = Groq(api_key=GROQ_API_KEY)
                        print(">>> Key updated! Retrying same row...")
                        continue
                    else:
                        import sys; sys.exit(1)
                
                print(f"\n  [Error] {err_str[:120]}")
                time.sleep(15)

            if retries >= 5:
                print("  [Skip] Failed 5 times.")
                raw_response = f"ERROR after retries: {err_str[:100]}"

    new_row = list(row) + [inf_valence, inf_arousal, raw_response]

    pd.DataFrame([new_row]).to_csv(
        OUTPUT_FILE,
        mode='a',
        header=False,
        index=False
    )


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\nDone. Results saved to {OUTPUT_FILE}")

df_out = pd.read_csv(OUTPUT_FILE)

df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'],errors='coerce')
df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'],errors='coerce')

valid = df_out.dropna(subset=['inferred_valence','inferred_arousal'])
failures = len(df_out) - len(valid)

print(f"Valid predictions: {len(valid)} / {len(df_out)}")
print(f"Failures / nulls: {failures}")