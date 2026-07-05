# import pandas as pd
# import os

# def clean_id(series):
#     return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# # 1. Load the LLM results
# df = pd.read_csv('llm_results_final2.csv')
# df['clip_id'] = clean_id(df['clip_title'])

# # 2. Generate Inferred Valence Summary
# v_summary = df.pivot(index='clip_id', columns='participant', values='inferred_valence')
# v_summary['avg_inferred_valence'] = v_summary.mean(axis=1)
# v_summary.to_csv('inferred_valence_summary.csv')

# # 3. Generate Inferred Arousal Summary
# a_summary = df.pivot(index='clip_id', columns='participant', values='inferred_arousal')
# a_summary['avg_inferred_arousal'] = a_summary.mean(axis=1)
# a_summary.to_csv('inferred_arousal_summary.csv')

# print("Success: 'inferred_valence_summary.csv' and 'inferred_arousal_summary.csv' created.")

import pandas as pd
import os

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# 1. Load the LLM results
if not os.path.exists('llm_results_final2.csv'):
    print("Error: llm_results_final2.csv not found!")
else:
    df = pd.read_csv('llm_results_final2.csv')
    df['clip_id'] = clean_id(df['clip_title'])

    # 2. FORCE NUMERIC (The Fix for your Error)
    # This converts "3" to 3 and turns any errors into NaN so the mean() works
    df['inferred_valence'] = pd.to_numeric(df['inferred_valence'], errors='coerce')
    df['inferred_arousal'] = pd.to_numeric(df['inferred_arousal'], errors='coerce')

    # 3. Generate Inferred Valence Summary
    # We use pivot_table with mean in case there are duplicate rows for a user/clip
    v_summary = df.pivot_table(index='clip_id', columns='participant', values='inferred_valence', aggfunc='mean')
    v_summary['avg_inferred_valence'] = v_summary.mean(axis=1)
    v_summary.to_csv('inferred_valence_summary.csv')

    # 4. Generate Inferred Arousal Summary
    a_summary = df.pivot_table(index='clip_id', columns='participant', values='inferred_arousal', aggfunc='mean')
    a_summary['avg_inferred_arousal'] = a_summary.mean(axis=1)
    a_summary.to_csv('inferred_arousal_summary.csv')

    print("Success: Created 'inferred_valence_summary.csv' and 'inferred_arousal_summary.csv'")