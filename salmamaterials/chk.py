import pandas as pd

files = [
    'perceived_valence_summary.csv', 
    'induced_valence_summary.csv', 
    'inferred_valence_summary.csv'
]

for f in files:
    try:
        df = pd.read_csv(f)
        print(f"\nFile: {f}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Could not read {f}: {e}")