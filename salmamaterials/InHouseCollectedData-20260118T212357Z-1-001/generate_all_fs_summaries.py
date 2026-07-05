import pandas as pd
import numpy as np
import os

MODELS = {
    'llama': 'llm_results_llama_fewshot.csv',
    'qwen':  'llm_results_mistral_fewshot.csv',
    'llama4':'llm_results_llama4_fewshot.csv'
}

DATA_DIR = 'InHouseCollectedData'

def generate_summaries():
    for model_name, filename in MODELS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {model_name}: {filepath} not found.")
            continue
            
        print(f"Processing {model_name} from {filepath}...")
        df = pd.read_csv(filepath)
        
        for dim in ['valence', 'arousal']:
            # Group by clip_title and average
            summary = df.groupby('clip_title').agg({
                f'inferred_{dim}': 'mean'
            }).reset_index()
            
            summary.rename(columns={
                'clip_title': 'clip_id',
                f'inferred_{dim}': f'avg_inferred_{dim}'
            }, inplace=True)
            
            # Clean ID (remove .mp4)
            summary['clip_id'] = summary['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
            
            output_name = f'inferred_{dim}_summary_{model_name}_FS.csv'
            summary.to_csv(output_name, index=False)
            print(f"✅ Generated {output_name}")

if __name__ == "__main__":
    generate_summaries()
