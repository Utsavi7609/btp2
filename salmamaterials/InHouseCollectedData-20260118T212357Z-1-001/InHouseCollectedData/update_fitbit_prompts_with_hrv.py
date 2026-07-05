import pandas as pd
import numpy as np
import re
import os

INPUT_FILE = "fitbit_ready_for_llm.csv"
OUTPUT_FILE = "fitbit_ready_for_llm_v2.csv"

def compute_rmssd(bpm_list):
    if len(bpm_list) < 3:
        return 0.0
    bpm_array = np.array(bpm_list)
    # Filter out zeros or invalid BPM
    bpm_array = bpm_array[bpm_array > 0]
    if len(bpm_array) < 3:
        return 0.0
    
    rr = 60000.0 / bpm_array
    diff_rr = np.diff(rr)
    return np.sqrt(np.mean(diff_rr ** 2))

if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"Processing {len(df)} rows to inject RMSSD...")

def update_prompt(prompt_str):
    # Find the heart rate sequence
    match = re.search(r'\[Heart Rate Sequence \(BPM\)\]: \[(.*?)\]', prompt_str)
    if not match:
        return prompt_str
    
    bpm_str = match.group(1)
    try:
        bpm_list = [float(x.strip()) for x in bpm_str.split(',') if x.strip()]
    except:
        return prompt_str
    
    rmssd = compute_rmssd(bpm_list)
    
    # Inject into Statistical Summary
    new_stats_item = f", RMSSD = {rmssd:.2f} ms"
    
    # Find the end of Statistical Summary
    if "[Statistical Summary]: " in prompt_str:
        # Insert before the closing period of the summary
        parts = prompt_str.split("[Statistical Summary]: ")
        summary_part = parts[1].split("\n\n### Question:")[0]
        # Append RMSSD to the end of the summary line
        updated_summary = summary_part.rstrip(".") + new_stats_item + "."
        
        new_prompt = parts[0] + "[Statistical Summary]: " + updated_summary + "\n\n### Question:" + parts[1].split("\n\n### Question:")[1]
        return new_prompt
    
    return prompt_str

df['llm_question'] = df['llm_question'].apply(update_prompt)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Success! Saved {OUTPUT_FILE} with RMSSD features injected.")
