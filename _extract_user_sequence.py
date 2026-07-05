import sys
import subprocess

def ensure_dependencies():
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        print("Installing required packages (pandas, openpyxl)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"])
        
ensure_dependencies()
import pandas as pd

excel_path = r"d:\BTP\btp2\files\input_files\clip_responses_export.xlsx"
print(f"Reading {excel_path}...")

try:
    df = pd.read_excel(excel_path)
    
    # Verify created_at exists
    if 'created_at' not in df.columns:
        print("ERROR: 'created_at' column not found in the Excel file.")
        print("Columns found:", df.columns.tolist())
        sys.exit(1)
        
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Sort chronologically per user
    df = df.sort_values(['participant', 'created_at'])
    
    target_users = ['Nishant', 'debjit2001', 'pranjal123', 'aritra', 'sayantankuila', 'ajaycc17']
    chrono_result = {u: [] for u in target_users}
    
    for _, row in df.iterrows():
        p = row['participant']
        if p in target_users:
            # We want the clip_title or ID
            chrono_result[p].append(row['clip_title'])
            
    print("\nTotal clips found for each user based on chronological timestamps:")
    for u, seq in chrono_result.items():
        print(f"  {u}: {len(seq)} clips")

    # Pad sequences if some users didn't finish all clips
    max_len = max([len(seq) for seq in chrono_result.values()] + [0])
    if max_len == 0:
        print("No clips found for the target users.")
        sys.exit(1)
        
    df_dict = {}
    for user, seq in chrono_result.items():
        df_dict[user] = seq + [None] * (max_len - len(seq))
        
    out_df = pd.DataFrame(df_dict)
    out_file = r"d:\BTP\btp2\_chronological_clip_orders.csv"
    out_df.to_csv(out_file, index=False)
    
    print(f"\n=> SUCCESS! Created {out_file} containing the actual chronological order.")

except Exception as e:
    print(f"An error occurred: {e}")
