# import google.generativeai as genai
# import pandas as pd
# import time
# import os
# import json

# # ==========================================
# # STEP 1: PASTE YOUR KEY HERE
# # ==========================================
# GOOGLE_API_KEY = "AIzaSyBp-tlfJcr6nX5mL_yWLwCw3s_X2ywvjFc"
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')

# # ==========================================
# # STEP 2: LOAD DATA
# # ==========================================
# input_file = 'fitbit_ready_for_llm.csv'
# output_file = 'llm_results_final.csv'

# if not os.path.exists(input_file):
#     print(f"Error: Could not find {input_file} in this folder.")
#     exit()

# df = pd.read_csv(input_file)

# # If we already started, load the existing progress
# if os.path.exists(output_file):
#     df_results = pd.read_csv(output_file)
#     print(f"Resuming from row {len(df_results)}...")
# else:
#     df_results = pd.DataFrame()

# # ==========================================
# # STEP 3: START AUTOMATED PROCESSING
# # ==========================================
# start_index = len(df_results)

# for i in range(start_index, len(df)):
#     row = df.iloc[i]
#     print(f"Processing row {i+1}/{len(df)} (User: {row['participant']})...")
    
#     try:
#         # Ask the AI
#         response = model.generate_content(row['llm_question'])
        
#         # Clean the AI's response to extract only the JSON data
#         raw_text = response.text.strip()
#         clean_json = raw_text.replace('```json', '').replace('```', '').strip()
        
#         # Parse the JSON answer
#         ai_data = json.loads(clean_json)
        
#         # Create a new row with the results
#         new_entry = row.to_dict()
#         new_entry['inferred_valence'] = ai_data.get('valence')
#         new_entry['inferred_arousal'] = ai_data.get('arousal')
        
#         # Convert to DataFrame and append
#         df_current = pd.DataFrame([new_entry])
#         df_results = pd.concat([df_results, df_current], ignore_index=True)
        
#         # SAVE CHECKPOINT: Save every 10 rows so you don't lose progress
#         if i % 10 == 0:
#             df_results.to_csv(output_file, index=False)
            
#         # RATE LIMITING: The free version of Gemini allows 15 requests per minute.
#         # We wait 4 seconds between rows to stay safe.
#         time.sleep(4)

#     except Exception as e:
#         print(f"!!! Error on row {i}: {e}")
#         print("Waiting 30 seconds before trying the next row...")
#         time.sleep(30)

# # Final Save
# df_results.to_csv(output_file, index=False)
# print(f"SUCCESS! All results saved to {output_file}")


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
# STEP 2: SETUP
# ==========================================
input_file = 'fitbit_ready_for_llm.csv'
output_file = 'llm_results_final.csv'

df = pd.read_csv(input_file)

# Resume from where you left off
if os.path.exists(output_file):
    df_results = pd.read_csv(output_file)
    start_index = len(df_results)
    print(f"Resuming from row {start_index}...")
else:
    df_results = pd.DataFrame()
    start_index = 0

# ==========================================
# STEP 3: THE AUTOMATED LOOP
# ==========================================
print(f"Starting processing with 'gemini-flash-latest'...")

for i in range(start_index, len(df)):
    row = df.iloc[i]
    print(f"Row {i+1}/{len(df)} | User: {row['participant']} | Clip: {row['clip_title']}...", end=" ", flush=True)
    
    success = False
    attempts = 0
    
    while not success and attempts < 3:
        try:
            # We use the name confirmed in your check_models list
            response = client.models.generate_content(
                model='gemini-flash-latest', 
                contents=row['llm_question']
            )
            
            # Clean and parse the JSON answer
            raw_text = response.text.strip()
            clean_json = raw_text.replace('```json', '').replace('```', '').strip()
            ai_data = json.loads(clean_json)
            
            # Build the result row
            new_entry = row.to_dict()
            new_entry['inferred_valence'] = ai_data.get('valence')
            new_entry['inferred_arousal'] = ai_data.get('arousal')
            
            # Save the result immediately
            df_results = pd.concat([df_results, pd.DataFrame([new_entry])], ignore_index=True)
            df_results.to_csv(output_file, index=False)
            
            print("Done.")
            success = True
            # Safety delay: 10 seconds to stay within free-tier limits
            time.sleep(10) 

        except Exception as e:
            attempts += 1
            if "429" in str(e):
                print(f"\n   [Rate Limit] Waiting 65s (Attempt {attempts}/3)...")
                time.sleep(65)
            else:
                print(f"\n   [Error] {e}")
                time.sleep(5)
                if attempts >= 3:
                    print("   [Skip] Failed 3 times. Skipping row.")
                    success = True

print(f"\nFINISH! All results saved to {output_file}")