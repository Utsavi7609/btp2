import pandas as pd
import numpy as np
import os

# Files to summmarize
INPUT_FILES = {
    'valence': 'llm_results_llama_fewshot.csv',
    'arousal': 'llm_results_llama_fewshot.csv'
}

def clean_id(val):
    return str(val).split('.')[0].strip()

for dim, filename in INPUT_FILES.items():
    if not os.path.exists(filename):
        print(f"Skipping {dim}: {filename} not found.")
        continue
    
    df = pd.read_csv(filename)
    
    # Groups by clip_title and calculates average
    summary = df.groupby('clip_title').agg({
        f'inferred_{dim}': 'mean'
    }).reset_index()
    
    # Rename to match run_all_thresholds.py expectations
    summary.rename(columns={
        'clip_title': 'clip_id',
        f'inferred_{dim}': f'avg_inferred_{dim}'
    }, inplace=True)
    
    output_filename = f'inferred_{dim}_summary_FS.csv'
    summary.to_csv(output_filename, index=False)
    print(f"✅ Generated {output_filename} ({len(summary)} clips)")
