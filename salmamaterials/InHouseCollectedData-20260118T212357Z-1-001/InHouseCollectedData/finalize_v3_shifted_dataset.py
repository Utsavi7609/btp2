import pandas as pd
import numpy as np
import re
import os

# --- CONFIGURE ---
READY_CSV = "fitbit_ready_for_llm.csv"
CONTEXT_CSV = "fitbit_hr_context.csv"
OUTPUT_CSV = "fitbit_ready_for_llm_v3_shifted.csv"

def compute_rmssd(bpm_list):
    if len(bpm_list) < 3:
        return 0.0
    bpm_array = np.array(bpm_list)
    bpm_array = bpm_array[bpm_array > 0]
    if len(bpm_array) < 3:
        return 0.0
    
    rr = 60000.0 / bpm_array
    diff_rr = np.diff(rr)
    return np.sqrt(np.mean(diff_rr ** 2))

def update_prompt_with_new_data(prompt_str, new_bpm_list):
    # 1. Replace the [Heart Rate Sequence (BPM)]: [...] section
    new_bpm_str = ", ".join([str(int(x)) for x in new_bpm_list])
    prompt_str = re.sub(r'\[Heart Rate Sequence \(BPM\)\]: \[.*?\]', f'[Heart Rate Sequence (BPM)]: [{new_bpm_str}]', prompt_str)
    
    # 2. Update Statistical Summary (Avg BPM, BPM Drift, Max BPM)
    avg_bpm = round(sum(new_bpm_list)/len(new_bpm_list), 2)
    bpm_drift = int(new_bpm_list[-1] - new_bpm_list[0])
    max_bpm = int(max(new_bpm_list))
    rmssd = compute_rmssd(new_bpm_list)
    
    new_stats = f"Average BPM = {avg_bpm}, BPM Drift = {bpm_drift}, Max BPM = {max_bpm}, RMSSD = {rmssd:.2f} ms"
    
    # Replace the Statistical Summary line
    prompt_str = re.sub(r'\[Statistical Summary\]: .*?\.', f'[Statistical Summary]: {new_stats}.', prompt_str)
    
    return prompt_str

def finalize_dataset():
    if not os.path.exists(READY_CSV) or not os.path.exists(CONTEXT_CSV):
        print("Missing required CSV files.")
        return

    df_ready = pd.read_csv(READY_CSV)
    df_context = pd.read_csv(CONTEXT_CSV)
    
    # Convert 'hr_sequence' from string back to list of floats
    def parse_seq(seq_str):
        if isinstance(seq_str, str):
            # Sequence is stored like '[72, 73, ...]' or '72, 73, ...'
            clean = seq_str.strip('[]')
            return [float(x.strip()) for x in clean.split(',') if x.strip()]
        return []

    df_context['bpm_list'] = df_context['hr_sequence'].apply(parse_seq)
    
    # Merge
    # We use a left outer join to keep all 1,495 rows, updating only those where we found context.
    df_final = pd.merge(df_ready, df_context[['participant', 'clip_title', 'bpm_list']], on=['participant', 'clip_title'], how='left')
    
    print(f"Merging {len(df_ready)} rows with {len(df_context)} context rows...")
    
    count = 0
    for idx, row in df_final.iterrows():
        if isinstance(row['bpm_list'], list) and len(row['bpm_list']) > 0:
            df_final.at[idx, 'llm_question'] = update_prompt_with_new_data(row['llm_question'], row['bpm_list'])
            count += 1
            
    # Drop temp column and save
    df_final.drop(columns=['bpm_list'], inplace=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Success! Generated {OUTPUT_CSV} with {count} updated prompts using SHIFTED window data.")

if __name__ == "__main__":
    finalize_dataset()
