"""
run_groq_mistral.py
===================
INNER DIRECTORY: InHouseCollectedData/

Runs LLM inference using Mixtral-8x7B via Groq API.
Uses zero-shot prompting strategy from Health-LLM (Kim et al., 2024).

StressLLM (Thapa et al., 2025) uses BioMistralDARE as their Mistral variant.
Here we use Mixtral-8x7B-32768 which is available on Groq API and is the
general Mistral MoE model referenced in both papers for comparison.

NOTE: If your mentor specifically means BioMistral (biomedical fine-tuned),
that requires HuggingFace inference — see comment at bottom of this file.

Input:  fitbit_ready_for_llm.csv
Output: llm_results_mistral.csv
"""

import pandas as pd
from groq import Groq
import time
import json
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
MODEL        = "mixtral-8x7b-32768"   # Mistral MoE model on Groq
INPUT_FILE   = "fitbit_ready_for_llm.csv"
OUTPUT_FILE  = "llm_results_mistral.csv"

TEMPERATURE  = 0.0
MAX_TOKENS   = 60

# ─── SETUP ────────────────────────────────────────────────────────────────────
client   = Groq(api_key=GROQ_API_KEY)
df_all   = pd.read_csv(INPUT_FILE)

# Resume logic
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
                response_format={"type": "json_object"},
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_response = completion.choices[0].message.content.strip()
            res          = json.loads(raw_response)

            inf_valence  = res.get('valence')
            inf_arousal  = res.get('arousal')

            print(f"V={inf_valence}  A={inf_arousal}")
            time.sleep(1.2)
            break

        except Exception as e:
            retries += 1
            err_str = str(e)

            if "429" in err_str or "rate" in err_str.lower():
                wait = 45 * retries
                print(f"\n  [Rate limit] Waiting {wait}s (attempt {retries}/5)...")
                time.sleep(wait)
            elif "json" in err_str.lower():
                print(f"\n  [JSON parse error] Raw: {raw_response}")
                print(f"  Retrying... (attempt {retries}/5)")
                time.sleep(10)
            else:
                print(f"\n  [Error] {err_str}")
                time.sleep(15)

            if retries >= 5:
                print(f"  [Skip] Failed 5 times. Recording null.")
                raw_response = f"ERROR after 5 retries: {err_str}"

    new_row = list(row) + [inf_valence, inf_arousal, raw_response]
    pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")

df_out = pd.read_csv(OUTPUT_FILE)
df_out['inferred_valence'] = pd.to_numeric(df_out['inferred_valence'], errors='coerce')
df_out['inferred_arousal'] = pd.to_numeric(df_out['inferred_arousal'], errors='coerce')
valid    = df_out.dropna(subset=['inferred_valence', 'inferred_arousal'])
failures = len(df_out) - len(valid)
print(f"   Valid predictions: {len(valid)} / {len(df_out)}")
print(f"   Failures / nulls:  {failures}")


# ─── NOTE: BioMistral (HuggingFace) alternative ───────────────────────────────
# If mentor means BioMistral-7B-DARE (Labrak et al., 2024) specifically:
#
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
#
# model_name = "BioMistral/BioMistral-7B-DARE"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# # Then iterate through prompts similarly
#
# StressLLM (Thapa et al., 2025) Table III shows BioMistralDARE achieves
# MAE=0.80 on PMData stress prediction. Mixtral-8x7B via Groq is a
# stronger general-purpose alternative available without GPU setup.
