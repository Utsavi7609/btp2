# import pandas as pd
# import json
# import os
# import glob
# from datetime import datetime, timedelta

# # --- SETTINGS ---
# # Path to your top-level folder containing all Fitbit data
# FITBIT_ROOT_DIR = 'Fitbit_Data_Folder' 
# # The master file from Phase 1/2
# MASTER_DATA_FILE = 'clip_responses_export.xlsx'
# TARGET_USERS = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']

# def find_hr_file_recursively(root_dir, date_str):
#     """
#     Recursively searches for a heart_rate file matching the date.
#     Pattern: heart_rate-YYYY-MM-DD.json
#     """
#     search_pattern = os.path.join(root_dir, "**", f"heart_rate-{date_str}.json")
#     files = glob.glob(search_pattern, recursive=True)
#     return files[0] if files else None

# def extract_bpm_window(file_path, start_time_str):
#     """
#     Extracts BPM values for a 120-second window.
#     """
#     start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
#     end_dt = start_dt + timedelta(seconds=120)
    
#     with open(file_path, 'r') as f:
#         data = json.load(f)
    
#     readings = []
#     for entry in data:
#         # Note: Fitbit JSON timestamps can vary; adjust format if needed
#         # Standard format is 'MM/DD/YY HH:MM:SS'
#         entry_time = datetime.strptime(entry['dateTime'], '%m/%d/%y %H:%M:%S')
#         if start_dt <= entry_time <= end_dt:
#             readings.append(entry['value']['bpm'])
#     return readings

# # --- MAIN EXECUTION ---
# df_master = pd.read_excel(MASTER_DATA_FILE)
# results = []

# print("Searching for physiological signals...")

# for _, row in df_master.iterrows():
#     user = row['participant']
#     if user in TARGET_USERS:
#         timestamp = row['created_at']
#         date_part = timestamp.split(' ')[0] # Extract YYYY-MM-DD
        
#         # 1. Search for the file anywhere in the user's data
#         hr_file = find_hr_file_recursively(FITBIT_ROOT_DIR, date_part)
        
#         if hr_file:
#             bpm_readings = extract_bpm_window(hr_file, timestamp)
#             if bpm_readings:
#                 # Calculate basic stats as per Health-LLM paper requirements
#                 results.append({
#                     'participant': user,
#                     'clip_title': row['clip_title'],
#                     'timestamp': timestamp,
#                     'avg_bpm': round(sum(bpm_readings)/len(bpm_readings), 2),
#                     'bpm_drift': bpm_readings[-1] - bpm_readings[0],
#                     'max_bpm': max(bpm_readings),
#                     'hr_sequence': bpm_readings,
#                     'self_reported_valence': row['impact_valence'],
#                     'self_reported_arousal': row['impact_arousal']
#                 })

# # Save the unified dataset
# output_df = pd.DataFrame(results)
# output_df.to_csv('fitbit_hr_context.csv', index=False)
# print(f"Phase 3 Complete: Extracted signals for {len(results)} sessions into 'fitbit_hr_context.csv'.")

import pandas as pd
import json
import os
import glob
from datetime import datetime, timedelta

# --- SETTINGS ---
# Path to your top-level folder containing all Fitbit data
# '.' means it will search in the current folder
FITBIT_ROOT_DIR = '.' 
MASTER_DATA_FILE = 'clip_responses_export.xlsx'
TARGET_USERS = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']

def find_hr_file_recursively(root_dir, date_str):
    search_pattern = os.path.join(root_dir, "**", f"heart_rate-{date_str}.json")
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None

def extract_bpm_window(file_path, start_dt):
    # start_dt is the 'Submission Time'. 
    # We look 120 seconds BACKWARD to capture the stimulus period.
    window_start = start_dt - timedelta(seconds=120)
    window_end = start_dt
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    
    readings = []
    for entry in data:
        # Fitbit JSON format is usually 'MM/DD/YY HH:MM:SS'
        entry_time = datetime.strptime(entry['dateTime'], '%m/%d/%y %H:%M:%S')
        if window_start <= entry_time <= window_end:
            readings.append(entry['value']['bpm'])
    return readings

# --- MAIN EXECUTION ---
# Ensure openpyxl is installed: pip install openpyxl
df_master = pd.read_excel(MASTER_DATA_FILE)
results = []

print("Searching for physiological signals...")

for _, row in df_master.iterrows():
    user = row['participant']
    if user in TARGET_USERS:
        timestamp = row['created_at']
        
        # FIX: Use strftime instead of split for Timestamp objects
        date_part = timestamp.strftime('%Y-%m-%d')
        
        hr_file = find_hr_file_recursively(FITBIT_ROOT_DIR, date_part)
        
        if hr_file:
            bpm_readings = extract_bpm_window(hr_file, timestamp)
            if bpm_readings:
                results.append({
                    'participant': user,
                    'clip_title': row['clip_title'],
                    'timestamp': timestamp,
                    'avg_bpm': round(sum(bpm_readings)/len(bpm_readings), 2),
                    'bpm_drift': bpm_readings[-1] - bpm_readings[0],
                    'max_bpm': max(bpm_readings),
                    'hr_sequence': bpm_readings,
                    'self_reported_valence': row['impact_valence'],
                    'self_reported_arousal': row['impact_arousal']
                })

if results:
    output_df = pd.DataFrame(results)
    output_df.to_csv('fitbit_hr_context.csv', index=False)
    print(f"Phase 3 Complete: Extracted signals for {len(results)} sessions into 'fitbit_hr_context.csv'.")
else:
    print("No matching heart rate data found. Check if the Fitbit JSON files are in the subfolders.")