# """
# Extract Heart Rate data from Fitbit exports
# WINDOWS VERSION with CORRECT folder structure
# Handles: Physical Activity_GoogleData and Global Export Data
# """

# import pandas as pd
# import json
# from pathlib import Path
# from datetime import datetime, timedelta
# import numpy as np

# def load_heart_rate_json(user_folder):
#     """
#     Load HR data from JSON files
#     CORRECTED PATHS:
#     - Takeout/Fitbit/Physical Activity_GoogleData/heart_rate*.json
#     - Takeout/Fitbit/Global Export Data/heart_rate*.json
#     """
#     user_name = user_folder.name
#     all_hr_data = []
    
#     # CORRECTED search patterns
#     search_patterns = [
#         'Takeout/Fitbit/Physical Activity_GoogleData/heart_rate*.json',
#         'Takeout/Fitbit/Global Export Data/heart_rate*.json',
#     ]
    
#     for pattern in search_patterns:
#         pattern_parts = pattern.split('/')
        
#         # Manually construct the search
#         # for takeout_folder in user_folder.glob('takeout-*'):
#         #     fitbit_path = takeout_folder / 'Takeout' / 'Fitbit'
            
#         #     if not fitbit_path.exists():
#         #         continue
#         takeout_folders = []

#         # Case 1: takeout-xxxx folders
#         takeout_folders.extend(user_folder.glob('takeout-*'))

#         # Case 2: direct Takeout folder
#         direct_takeout = user_folder / 'Takeout'
#         if direct_takeout.exists():
#             takeout_folders.append(direct_takeout)

#         for takeout_folder in takeout_folders:

#             # handle both structures
#             if takeout_folder.name.lower().startswith("takeout"):
#                 fitbit_path = takeout_folder / 'Fitbit'
#             else:
#                 fitbit_path = takeout_folder / 'Takeout' / 'Fitbit'

#             if not fitbit_path.exists():
#                 continue    
            
#             # Check Physical Activity_GoogleData (note the underscore!)
#             physical_activity_path = fitbit_path / 'Physical Activity_GoogleData'
#             if physical_activity_path.exists():
#                 for json_file in physical_activity_path.glob('heart_rate*.json'):
#                     all_hr_data.extend(parse_hr_json(json_file, user_name))
            
#             # Check Global Export Data
#             global_export_path = fitbit_path / 'Global Export Data'
#             if global_export_path.exists():
#                 for json_file in global_export_path.glob('heart_rate*.json'):
#                     all_hr_data.extend(parse_hr_json(json_file, user_name))
    
#     if not all_hr_data:
#         return pd.DataFrame()
    
#     df = pd.DataFrame(all_hr_data)
#     df = df.sort_values('datetime').reset_index(drop=True)
    
#     return df

# def parse_hr_json(json_file, user_name):
#     """
#     Parse a single HR JSON file
#     """
#     print(f"  📄 Reading: {json_file.name}")
    
#     hr_data = []
    
#     try:
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if not data or not isinstance(data, list):
#             return hr_data
        
#         for entry in data:
#             try:
#                 dt_str = entry.get('dateTime')
                
#                 # Handle different value formats
#                 value = entry.get('value', {})
                
#                 # Format 1: {"bpm": 72, "confidence": 2}
#                 if isinstance(value, dict) and 'bpm' in value:
#                     bpm = value['bpm']
#                 # Format 2: {"resting": 65}
#                 elif isinstance(value, dict) and 'resting' in value:
#                     bpm = value['resting']
#                 # Format 3: Direct number
#                 elif isinstance(value, (int, float)):
#                     bpm = value
#                 else:
#                     continue
                
#                 if dt_str and bpm:
#                     # Parse datetime (format: MM/DD/YY HH:MM:SS)
#                     dt = datetime.strptime(dt_str, '%m/%d/%y %H:%M:%S')
                    
#                     hr_data.append({
#                         'datetime': dt,
#                         'heart_rate': float(bpm),
#                         'user_id': user_name
#                     })
#             except Exception as e:
#                 continue  # Skip malformed entries
                
#     except Exception as e:
#         print(f"    ⚠️  Error reading {json_file.name}: {e}")
    
#     return hr_data

# def create_hr_windows(hr_df, window_size_hours=24):
#     """
#     Create sliding windows of HR data for emotion inference
#     Args:
#         hr_df: DataFrame with heart rate data
#         window_size_hours: Size of each window in hours (default: 24 hours/1 day)
    
#     Returns:
#         List of windows, each containing HR statistics
#     """
#     if hr_df.empty:
#         return []
    
#     windows = []
#     user_id = hr_df['user_id'].iloc[0]
    
#     # Get date range
#     start_date = hr_df['datetime'].min().date()
#     end_date = hr_df['datetime'].max().date()
    
#     current_date = start_date
#     while current_date <= end_date:
#         # Define window
#         window_start = datetime.combine(current_date, datetime.min.time())
#         window_end = window_start + timedelta(hours=window_size_hours)
        
#         # Extract data in window
#         mask = (hr_df['datetime'] >= window_start) & (hr_df['datetime'] < window_end)
#         window_data = hr_df[mask]
        
#         if len(window_data) > 0:
#             hr_values = window_data['heart_rate'].values
            
#             window_info = {
#                 'user_id': user_id,
#                 'window_start': window_start,
#                 'window_end': window_end,
#                 'date': current_date.strftime('%Y-%m-%d'),
#                 'hr_values': hr_values,
#                 'hr_mean': np.mean(hr_values),
#                 'hr_std': np.std(hr_values),
#                 'hr_min': np.min(hr_values),
#                 'hr_max': np.max(hr_values),
#                 'hr_count': len(hr_values)
#             }
            
#             windows.append(window_info)
        
#         # Move to next day
#         current_date += timedelta(days=1)
    
#     return windows

# def main():
#     base_path = Path('InHouseCollectedData')
    
#     if not base_path.exists():
#         print(f"❌ Error: {base_path} not found!")
#         print(f"   Current directory: {Path.cwd()}")
#         return
    
#     all_hr_data = []
#     all_windows = []
    
#     for user_folder in base_path.iterdir():
#         if not user_folder.is_dir():
#             continue
        
#         print(f"\n🔍 Processing: {user_folder.name}")
        
#         # Load HR data
#         hr_df = load_heart_rate_json(user_folder)
#         if not hr_df.empty:
#             print(f"  ✅ Loaded {len(hr_df)} HR measurements")
#             all_hr_data.append(hr_df)
            
#             # Create windows for this user
#             windows = create_hr_windows(hr_df)
#             all_windows.extend(windows)
#             print(f"  ✅ Created {len(windows)} daily windows")
#         else:
#             print(f"  ⚠️  No HR data found")
    
#     # Save all data
#     if all_hr_data:
#         combined_hr = pd.concat(all_hr_data, ignore_index=True)
#         combined_hr.to_csv('fitbit_hr_raw.csv', index=False)
#         print(f"\n✅ Saved {len(combined_hr)} HR measurements to: fitbit_hr_raw.csv")
#     else:
#         print("\n❌ No HR data found!")
#         print("\n📁 Expected folder structure:")
#         print("   InHouseCollectedData/{{user}}/takeout-*/Takeout/Fitbit/Physical Activity_GoogleData/")
#         print("   InHouseCollectedData/{{user}}/takeout-*/Takeout/Fitbit/Global Export Data/")
    
#     if all_windows:
#         # Create windows DataFrame
#         windows_df = pd.DataFrame([
#             {
#                 'user_id': w['user_id'],
#                 'date': w['date'],
#                 'hr_values': ','.join(map(str, w['hr_values'][:50])),  # First 50 values
#                 'hr_mean': w['hr_mean'],
#                 'hr_std': w['hr_std'],
#                 'hr_min': w['hr_min'],
#                 'hr_max': w['hr_max'],
#                 'hr_count': w['hr_count']
#             }
#             for w in all_windows
#         ])
#         windows_df.to_csv('fitbit_hr_windows.csv', index=False)
#         print(f"✅ Saved {len(windows_df)} daily windows to: fitbit_hr_windows.csv")

# if __name__ == "__main__":
#     main()



"""
Extract Heart Rate data from Fitbit exports
Handles all Fitbit Takeout structures
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


def compute_rmssd(hr_values):

    if len(hr_values) < 3:
        return np.nan

    rr = 60000 / hr_values
    diff_rr = np.diff(rr)

    return np.sqrt(np.mean(diff_rr ** 2))


def load_heart_rate_json(user_folder):

    user_name = user_folder.name
    all_hr_data = []

    # search for ANY heart_rate json inside Fitbit folders
    for json_file in user_folder.rglob("heart_rate*.json"):

        if "Fitbit" not in str(json_file):
            continue

        all_hr_data.extend(parse_hr_json(json_file, user_name))

    if not all_hr_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_hr_data)

    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.drop_duplicates(subset=["datetime", "user_id"])

    return df


def parse_hr_json(json_file, user_name):

    print(f"  📄 Reading: {json_file.name}")

    hr_data = []

    try:

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return hr_data

        for entry in data:

            dt_str = entry.get("dateTime")
            value = entry.get("value", {})

            if isinstance(value, dict) and "bpm" in value:
                bpm = value["bpm"]

            elif isinstance(value, dict) and "resting" in value:
                bpm = value["resting"]

            elif isinstance(value, (int, float)):
                bpm = value

            else:
                continue

            if dt_str and bpm:

                dt = datetime.strptime(dt_str, "%m/%d/%y %H:%M:%S")

                hr_data.append(
                    {
                        "datetime": dt,
                        "heart_rate": float(bpm),
                        "user_id": user_name,
                    }
                )

    except Exception as e:
        print("Error:", e)

    return hr_data


def create_hr_windows(hr_df, window_size_hours=2):

    if hr_df.empty:
        return []

    windows = []

    user_id = hr_df["user_id"].iloc[0]

    start = hr_df["datetime"].min()
    end = hr_df["datetime"].max()

    current = start

    while current < end:

        window_end = current + timedelta(hours=window_size_hours)

        mask = (hr_df["datetime"] >= current) & (hr_df["datetime"] < window_end)

        window_data = hr_df[mask]

        if len(window_data) > 0:

            hr_values = window_data["heart_rate"].values

            windows.append(
                {
                    "user_id": user_id,
                    "date": current.strftime("%Y-%m-%d %H:%M"),
                    "hr_mean": np.mean(hr_values),
                    "hr_std": np.std(hr_values),
                    "hr_min": np.min(hr_values),
                    "hr_max": np.max(hr_values),
                    "hr_count": len(hr_values),
                    "hrv_rmssd": compute_rmssd(hr_values),
                }
            )

        current = window_end

    return windows


def main():

    base_path = Path("InHouseCollectedData")

    all_hr_data = []
    all_windows = []

    for user_folder in base_path.iterdir():

        if not user_folder.is_dir():
            continue

        print(f"\nProcessing: {user_folder.name}")

        hr_df = load_heart_rate_json(user_folder)

        if not hr_df.empty:

            print(f"  Loaded {len(hr_df)} HR measurements")

            all_hr_data.append(hr_df)

            windows = create_hr_windows(hr_df)

            all_windows.extend(windows)

            print(f"  Created {len(windows)} windows")

        else:
            print("  No HR data found")

    if all_hr_data:

        combined_hr = pd.concat(all_hr_data, ignore_index=True)

        combined_hr.to_csv("fitbit_hr_raw.csv", index=False)

        print(f"\nSaved {len(combined_hr)} HR measurements")

    if all_windows:

        windows_df = pd.DataFrame(all_windows)

        windows_df.to_csv("fitbit_hr_windows.csv", index=False)

        print(f"Saved {len(windows_df)} windows")


if __name__ == "__main__":
    main()