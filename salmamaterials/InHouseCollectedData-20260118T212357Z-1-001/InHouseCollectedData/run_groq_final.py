import pandas as pd
from groq import Groq
import time, json, os

# Get your key from console.groq.com
client = Groq(api_key="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")
input_file = "fitbit_ready_for_llm.csv"
output_file = "llm_results_final2.csv"

df_all = pd.read_csv(input_file)

# Resume logic: checks existing count
if os.path.exists(output_file):
    start_row = len(pd.read_csv(output_file))
    print(f"Resuming from row {start_row}...")
else:
    start_row = 0
    pd.DataFrame(columns=list(df_all.columns) + ['inferred_valence', 'inferred_arousal']).to_csv(output_file, index=False)

for i in range(start_row, len(df_all)):
    row = df_all.iloc[i]
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": row['llm_question']}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        res = json.loads(completion.choices[0].message.content)
        
        # Save one row at a time so no data is lost
        new_data = pd.DataFrame([list(row) + [res.get('valence'), res.get('arousal')]])
        new_data.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Done {i+1}/{len(df_all)}: {row['clip_title']}")
        time.sleep(1.2) # Small delay for rate limits
    except Exception as e:
        print(f"Error at {i}: {e}")
        time.sleep(10)