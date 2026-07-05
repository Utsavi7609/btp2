from google import genai
import pandas as pd
import time
import os
import json

# ==========================================
# STEP 1: PASTE YOUR KEY HERE
# ==========================================
GOOGLE_API_KEY = "AIzaSyD51usgMoh7w7OlOaon-e8xdZ8Y6RiBfUU"
client = genai.Client(api_key=GOOGLE_API_KEY)

# ==========================================
# STEP 2: SETUP
# ==========================================
input_file = 'fitbit_ready_for_llm.csv'
output_file = 'llm_results_final2.csv'

df = pd.read_csv(input_file)

if os.path.exists(output_file):
    df_results = pd.read_csv(output_file)
    start_index = len(df_results)
    print(f"Resuming from row {start_index}...")
else:
    df_results = pd.DataFrame()
    start_index = 0

# ==========================================
# STEP 3: THE "STUBBORN" LOOP
# ==========================================
print(f"Starting run with 'gemini-flash-latest'...")

for i in range(start_index, len(df)):
    row = df.iloc[i]
    print(f"Row {i+1}/{len(df)} | {row['participant']} | {row['clip_title']}...", end=" ", flush=True)
    
    success = False
    while not success:
        try:
            # Reverted to the name that worked for your Row 1-5
            response = client.models.generate_content(
                model='gemini-flash-latest', 
                contents=row['llm_question']
            )
            
            raw_text = response.text.strip()
            clean_json = raw_text.replace('```json', '').replace('```', '').strip()
            ai_data = json.loads(clean_json)
            
            new_entry = row.to_dict()
            new_entry['inferred_valence'] = ai_data.get('valence')
            new_entry['inferred_arousal'] = ai_data.get('arousal')
            
            # Save immediately so no data is lost
            df_results = pd.concat([df_results, pd.DataFrame([new_entry])], ignore_index=True)
            df_results.to_csv(output_file, index=False)
            
            print("Done.")
            success = True
            # Wait 20 seconds to be extremely safe from rate limits
            time.sleep(45) 

        except Exception as e:
            if "429" in str(e):
                print(f"\n   [Rate Limit] Server busy. Waiting 90 seconds...")
                time.sleep(95)
            else:
                print(f"\n   [Error] {e}. Retrying in 10 seconds...")
                time.sleep(20)

print(f"\nFINISH! All results saved to {output_file}")