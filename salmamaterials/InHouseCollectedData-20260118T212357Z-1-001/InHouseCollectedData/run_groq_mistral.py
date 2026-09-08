# # """
# # run_groq_mistral.py
# # ===================
# # INNER DIRECTORY: InHouseCollectedData/

# # Runs LLM inference using Mixtral-8x7B via Groq API.
# # Uses zero-shot prompting strategy from Health-LLM (Kim et al., 2024).

# # StressLLM (Thapa et al., 2025) uses BioMistralDARE as their Mistral variant.
# # Here we use Mixtral-8x7B-32768 which is available on Groq API and is the
# # general Mistral MoE model referenced in both papers for comparison.

# # NOTE: If your mentor specifically means BioMistral (biomedical fine-tuned),
# # that requires HuggingFace inference — see comment at bottom of this file.

# # Input:  fitbit_ready_for_llm.csv
# # Output: llm_results_mistral.csv
# # """

# # import pandas as pd
# # from groq import Groq
# # import time
# # import json
# # import os

# # # ─── CONFIGURATION ────────────────────────────────────────────────────────────
# # GROQ_API_KEY = "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"
# # MODEL        = "mixtral-8x7b-32768"   # Mistral MoE model on Groq
# # INPUT_FILE   = "fitbit_ready_for_llm.csv"
# # OUTPUT_FILE  = "llm_results_mistral.csv"

# # TEMPERATURE  = 0.0
# # MAX_TOKENS   = 60

# # # ─── SETUP ────────────────────────────────────────────────────────────────────
# # client   = Groq(api_key=GROQ_API_KEY)
# # df_all   = pd.read_csv(INPUT_FILE)

# # # Resume logic
# # if os.path.exists(OUTPUT_FILE):
# #     df_done   = pd.read_csv(OUTPUT_FILE)
# #     start_row = len(df_done)
# #     print(f"Resuming from row {start_row} / {len(df_all)}")
# # else:
# #     start_row = 0
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
# #                 response_format={"type": "json_object"},
# #                 temperature=TEMPERATURE,
# #                 max_tokens=MAX_TOKENS,
# #             )
# #             raw_response = completion.choices[0].message.content.strip()
# #             res          = json.loads(raw_response)

# #             inf_valence  = res.get('valence')
# #             inf_arousal  = res.get('arousal')

# #             print(f"V={inf_valence}  A={inf_arousal}")
# #             time.sleep(1.2)
# #             break

# #         except Exception as e:
# #             retries += 1
# #             err_str = str(e)

# #             if "429" in err_str or "rate" in err_str.lower():
# #                 wait = 45 * retries
# #                 print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
# #                 time.sleep(wait)
# #             elif "json" in err_str.lower():
# #                 print(f"\n  [JSON parse error] Raw: {raw_response}")
# #                 print(f"  Retrying... (attempt {retries}/5)")
# #                 time.sleep(10)
# #             else:
# #                 print(f"\n  [Error] {err_str}")
# #                 time.sleep(15)

# #             if retries >= 5:
# #                 print(f"  [Skip] Failed 5 times. Recording null.")
# #                 raw_response = f"ERROR after 5 retries: {err_str}"

# #     new_row = list(row) + [inf_valence, inf_arousal, raw_response]
# #     pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

# # print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")

# # df_out = pd.read_csv(OUTPUT_FILE)
# # df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
# # df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')
# # valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
# # failures = len(df_out) - len(valid)
# # print(f"   Valid predictions: {len(valid)} / {len(df_out)}")
# # print(f"   Failures / nulls:  {failures}")


# # # ─── NOTE: BioMistral (HuggingFace) alternative ───────────────────────────────
# # # If mentor means BioMistral-7B-DARE (Labrak et al., 2024) specifically:
# # #
# # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # import torch
# # #
# # # model_name = "BioMistral/BioMistral-7B-DARE"
# # # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# # # # Then iterate through prompts similarly
# # #
# # # StressLLM (Thapa et al., 2025) Table III shows BioMistralDARE achieves
# # # MAE=0.80 on PMData stress prediction. Mixtral-8x7B via Groq is a
# # # stronger general-purpose alternative available without GPU setup.



# import pandas as pd
# from groq import Groq
# import time
# import json
# import os
# import random
# import re

# GROQ_API_KEY = "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"
# MODEL = "openai/gpt-oss-20b"
# # MODEL = "qwen/qwen3-32b"
# INPUT_FILE = "fitbit_ready_for_llm.csv"
# OUTPUT_FILE = "llm_results_mistral.csv"

# TEMPERATURE = 0.0
# MAX_TOKENS = 200

# client = Groq(api_key=GROQ_API_KEY)
# df_all = pd.read_csv(INPUT_FILE)

# # Resume logic
# if os.path.exists(OUTPUT_FILE):
#     df_done = pd.read_csv(OUTPUT_FILE)
#     start_row = len(df_done)
#     print(f"Resuming from row {start_row} / {len(df_all)}")
# else:
#     start_row = 0
#     header_cols = list(df_all.columns) + ['inferred_valence','inferred_arousal','raw_response']
#     pd.DataFrame(columns=header_cols).to_csv(OUTPUT_FILE,index=False)
#     print(f"Starting fresh. {len(df_all)} rows to process.")

# print(f"Model: {MODEL}\n")


# def extract_json(text):
#     """
#     Extract JSON from model output even if it contains extra text
#     """
#     try:
#         return json.loads(text)
#     except:
#         match = re.search(r"\{.*?\}", text, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except:
#                 return None
#     return None


# for i in range(start_row, len(df_all)):

#     row = df_all.iloc[i]

#     print(
#         f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
#         end="  ",
#         flush=True
#     )

#     inf_valence = None
#     inf_arousal = None
#     raw_response = ""
#     retries = 0

#     while retries < 5:

#         try:

#             # completion = client.chat.completions.create(
#             #     messages=[{"role": "user", "content": row['llm_question']}],
#             #     model=MODEL,
#             #     temperature=TEMPERATURE,
#             #     max_tokens=MAX_TOKENS
#             # )
#             completion = client.chat.completions.create(
#     messages=[{"role": "user", "content": row['llm_question']}],
#     model=MODEL,
#     temperature=TEMPERATURE,
#     max_tokens=200,
#     response_format={"type": "json_object"}
# )

#             raw_response = completion.choices[0].message.content.strip()

#             parsed = extract_json(raw_response)

#             if parsed:
#                 inf_valence = parsed.get("valence")
#                 inf_arousal = parsed.get("arousal")

#             print(f"V={inf_valence}  A={inf_arousal}")

#             time.sleep(1.6 + random.uniform(0,0.4))

#             break

#         except Exception as e:

#             retries += 1
#             err_str = str(e)

#             if "429" in err_str or "rate" in err_str.lower():

#                 wait = 60 * retries
#                 print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
#                 time.sleep(wait)

#             elif "json" in err_str.lower():

#                 print(f"\n  [JSON parse error] Raw: {raw_response}")
#                 print(f"  Retrying... (attempt {retries}/5)")
#                 time.sleep(10)

#             else:

#                 print(f"\n  [Error] {err_str}")
#                 time.sleep(15)

#             if retries >= 5:
#                 print("  [Skip] Failed 5 times. Recording null.")
#                 raw_response = f"ERROR after 5 retries: {err_str}"

#     new_row = list(row) + [inf_valence, inf_arousal, raw_response]

#     pd.DataFrame([new_row]).to_csv(
#         OUTPUT_FILE,
#         mode="a",
#         header=False,
#         index=False
#     )


# print(f"\nDone. Results saved to {OUTPUT_FILE}")

# df_out = pd.read_csv(OUTPUT_FILE)

# df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
# df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')

# valid = df_out.dropna(subset=['inferred_valence','inferred_arousal'])
# failures = len(df_out) - len(valid)

# print(f"Valid predictions: {len(valid)} / {len(df_out)}")
# print(f"Failures / nulls: {failures}")


"""
run_groq_mistral.py
===================
INNER DIRECTORY: InHouseCollectedData/

Second LLM for comparison against Llama-3.3-70B.
Uses qwen/qwen3-32b (32B parameters, available on your Groq account).

WHY PREVIOUS ATTEMPTS FAILED:
  - mixtral-8x7b-32768, mistral-saba-24b: decommissioned by Groq
  - mistral-large-latest: not hosted on Groq (Mistral's own API only)
  - qwen3-32b with response_format=json_object: Groq returns 400 error
  - qwen3-32b without json_object: model outputs <think>...</think> chain-of-thought
    BEFORE the actual JSON, so raw_response looks empty to a plain json.loads()

FIX:
  1. Add /no_think directive to the prompt → tells qwen3 to skip thinking mode
  2. Strip <think>...</think> blocks as fallback even if /no_think is ignored
  3. Do NOT use response_format={"type": "json_object"} (causes 400 on qwen3)

Input:  fitbit_ready_for_llm.csv
Output: llm_results_mistral.csv
"""

import pandas as pd
from groq import Groq
import time
import json
import os
import random
import re

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ["GROQ_API_KEY"]  # set this in your shell/env, do not hardcode

# 

MODEL        = "qwen/qwen3-32b"

INPUT_FILE   = "fitbit_ready_for_llm_v3_shifted.csv"
OUTPUT_FILE  = "llm_results_qwen_v3_shifted.csv"

TEMPERATURE  = 0.0
MAX_TOKENS   = 120

# ─── SETUP ────────────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)
df_all = pd.read_csv(INPUT_FILE)

# Auto-clean if previous run produced all nulls
if os.path.exists(OUTPUT_FILE):
    df_check = pd.read_csv(OUTPUT_FILE)
    df_check['inferred_valence'] = pd.to_numeric(df_check['inferred_valence'], errors='coerce')
    if df_check['inferred_valence'].isna().all() and len(df_check) > 0:
        print("Previous output file has all nulls — deleting and starting fresh.")
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

print(f"Model: {MODEL}")
print("Thinking mode disabled via /no_think directive\n")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def strip_thinking(text):
    """Remove <think>...</think> blocks that qwen3 sometimes emits."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_json_robust(text):
    """
    Extract valence/arousal from model output.
    Three fallback layers so we never miss a valid response.
    """
    text = strip_thinking(text)
    if not text:
        return None

    # Layer 1: whole text is valid JSON
    try:
        obj = json.loads(text)
        if 'valence' in obj and 'arousal' in obj:
            return obj
    except Exception:
        pass

    # Layer 2: find first { ... } block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if 'valence' in obj and 'arousal' in obj:
                return obj
        except Exception:
            pass

    # Layer 3: regex extract numbers
    v = re.search(r'["\s]?[Vv]alence["\s]*[:=]\s*([1-5])', text)
    a = re.search(r'["\s]?[Aa]rousal["\s]*[:=]\s*([1-5])', text)
    if v and a:
        return {'valence': int(v.group(1)), 'arousal': int(a.group(1))}

    return None


def build_prompt(base_prompt, row_idx, row_data, df):
    """
    Inject Few-Shot examples, append /no_think, and tighten the output instruction.
    Reverted to full sequence extraction for consistency across entire run.
    """
    valid_candidates = df[
        (df['participant'] != row_data['participant']) & 
        (df['participant_valence'].notna()) & 
        (df['participant_arousal'].notna())
    ]
    few_shot_examples = "### Here are a few reference examples mapping physiological data to true scores:\n\n"
    if len(valid_candidates) >= 3:
        sampled = valid_candidates.sample(3, random_state=row_idx)
        for _, ex_row in sampled.iterrows():
            match = re.search(r'(### Sensor Readings: .*?)\n\n### Question:', str(ex_row['llm_question']), re.DOTALL)
            ex_sensors = match.group(1).replace('### Sensor Readings: ', '').strip() if match else None
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

    # Replace the response instruction line to be maximally explicit
    final_prompt = re.sub(
        r'### Response:.*',
        '### Response: Output ONLY valid JSON, nothing else: {"valence": X, "arousal": Y}\n\n/no_think',
        injected_prompt,
        flags=re.DOTALL
    )
    return final_prompt


# ─── INFERENCE LOOP ───────────────────────────────────────────────────────────
for i in range(start_row, len(df_all)):
    row = df_all.iloc[i]
    print(
        f"[{i+1:04d}/{len(df_all)}] {row['participant']:15s} | {row['clip_title'][:35]:35s}",
        end="  ", flush=True
    )

    prompt_text  = build_prompt(str(row['llm_question']), i, row, df_all)
    inf_valence  = None
    inf_arousal  = None
    raw_response = ""
    retries      = 0

    while retries < 5:
        try:
            # No response_format parameter — qwen3 rejects json_object mode
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            raw_response = completion.choices[0].message.content.strip()
            parsed       = extract_json_robust(raw_response)

            if parsed:
                inf_valence = parsed.get('valence')
                inf_arousal = parsed.get('arousal')
                print(f"V={inf_valence}  A={inf_arousal}")
            else:
                cleaned = strip_thinking(raw_response)
                print(f"V=None  A=None  [got: {repr(cleaned[:80])}]")

            time.sleep(1.8)
            break

        except Exception as e:
            retries += 1
            err_str = str(e)

            if "429" in err_str or "rate" in err_str.lower():
                wait = 60 * retries
                print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
                time.sleep(wait)

            elif "decommissioned" in err_str or "model_not_found" in err_str:
                print(f"\n  [Model error] {MODEL} not available.")
                print("  Available models on your account:")
                print("    qwen/qwen3-32b, moonshotai/kimi-k2-instruct,")
                print("    openai/gpt-oss-20b, openai/gpt-oss-120b,")
                print("    meta-llama/llama-4-scout-17b-16e-instruct")
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

# Debug: show what failed rows actually returned
if failures > 0:
    print(f"\nSample of failed raw responses (to diagnose any remaining issues):")
    sample = df_out[df_out['inferred_valence'].isna()]['raw_response'].head(3)
    for idx, raw in sample.items():
        print(f"  Row {idx}: {repr(strip_thinking(str(raw))[:150])}")