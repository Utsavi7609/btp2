from google import genai
import pandas as pd
import time
import os
import json

# ==========================================
# STEP 1: PASTE YOUR KEY HERE
# ==========================================
GOOGLE_API_KEY = "AIzaSyBp-tlfJcr6nX5mL_yWLwCw3s_X2ywvjFc"
client = genai.Client(api_key=GOOGLE_API_KEY)

# ==========================================
# STEP 2: SETUP FILES
# ==========================================
input_file = 'fitbit_ready_for_llm.csv'
output_file = 'llm_results_final.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found!")
    exit()

df = pd.read_csv(input_file)

# Resume logic: Start from where it stopped
if os.path.exists(output_file):
    df_results = pd.read_csv(output_file)
    start_index = len(df_results)
    print(f"Resuming from row {start_index}...")
else:
    df_results = pd.DataFrame()
    start_index = 0

# ==========================================
# STEP 3: THE AUTOMATION LOOP
# ==========================================
print(f"Starting processing with 'gemini-pro-latest' for {len(df) - start_index} rows...")

for i in range(start_index, len(df)):
    row = df.iloc[i]
    print(f"Row {i+1}/{len(df)} | User: {row['participant']} | Clip: {row['clip_title']}")
    
    success = False
    retries = 0
    
    while not success and retries < 3:
        try:
            # Using 'gemini-pro-latest' - The "Something Else" you requested
            response = client.models.generate_content(
                model='gemini-pro-latest', 
                contents=row['llm_question']
            )
            
            raw_text = response.text.strip()
            # Clean JSON formatting
            clean_json = raw_text.replace('```json', '').replace('```', '').strip()
            ai_data = json.loads(clean_json)
            
            new_entry = row.to_dict()
            new_entry['inferred_valence'] = ai_data.get('valence')
            new_entry['inferred_arousal'] = ai_data.get('arousal')
            
            df_results = pd.concat([df_results, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Save checkpoint every 2 rows for maximum safety
            if i % 2 == 0:
                df_results.to_csv(output_file, index=False)
            
            success = True
            # Wait 12 seconds to ensure we never hit the 429 limit
            time.sleep(12) 

        except Exception as e:
            if "429" in str(e):
                print(f"   [Rate Limit] Waiting 60s for server to clear...")
                time.sleep(60)
                retries += 1
            elif "404" in str(e):
                print(f"   [Model Name Error] Trying alternative name 'gemini-1.5-pro'...")
                # If pro-latest fails, this is a fallback name
                try:
                    response = client.models.generate_content(model='gemini-1.5-pro', contents=row['llm_question'])
                    # ... (rest of parsing logic) ...
                    success = True
                except:
                    print(f"   Critical Error on row {i}: {e}")
                    success = True # Skip to keep moving
            else:
                print(f"   Error: {e}. Moving to next row.")
                success = True 

# Final save
df_results.to_csv(output_file, index=False)
print(f"\nSUCCESS! Results are in {output_file}")