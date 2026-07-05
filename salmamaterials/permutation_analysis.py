# import pandas as pd

# def run_analysis(dim, p_file, e_file, i_file, i_col):
#     # Load and clean data
#     def load(f):
#         df = pd.read_csv(f)
#         df['clip_id'] = df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
#         return df

#     p_df, e_df, i_df = load(p_file), load(e_file), load(i_file)
#     m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
#     m = pd.merge(m, i_df[['clip_id', i_col]], on='clip_id')
#     m.columns = ['id', 'p', 'e', 'i']

#     # Threshold match logic (abs diff <= 1)
#     def is_match(a, b): return abs(a - b) <= 1

#     results = {
#         "Q1 (Case 1.1): p=e=i": 0,
#         "Q2: p=i (but p!=e) Total": 0,
#         " - Case 2.1: e matches neither": 0,
#         " - Case 2.2: e matches i": 0,
#         "Q3: p=e (but p!=i) Total": 0,
#         " - Case 3.1: i matches neither": 0,
#         " - Case 3.2: i matches e": 0,
#         "No Match (Other permutations)": 0
#     }

#     for _, row in m.iterrows():
#         pe, pi, ei = is_match(row['p'], row['e']), is_match(row['p'], row['i']), is_match(row['e'], row['i'])

#         if pe and pi: # This captures p=e=i (and technically e=i follows)
#             results["Q1 (Case 1.1): p=e=i"] += 1
#         elif pi and not pe: # p=i but p!=e
#             results["Q2: p=i (but p!=e) Total"] += 1
#             if not ei:
#                 results[" - Case 2.1: e matches neither"] += 1
#             else:
#                 results[" - Case 2.2: e matches i"] += 1
#         elif pe and not pi: # p=e but p!=i
#             results["Q3: p=e (but p!=i) Total"] += 1
#             if not ei:
#                 results[" - Case 3.1: i matches neither"] += 1
#             else:
#                 results[" - Case 3.2: i matches e"] += 1
#         else:
#             results["No Match (Other permutations)"] += 1

#     return results

# # To run for VALENCE (Self-Reported)
# print("VALENCE PERMUTATION RESULTS (Self-Reported)")
# print("-" * 45)
# res_v = run_analysis('valence', 'perceived_valence_summary.csv', 'final_expressed_valence.csv', 'induced_valence_summary.csv', 'avg_valence')
# for k, v in res_v.items(): print(f"{k}: {v}")


import pandas as pd
import os

def run_analysis(label, dim, p_file, e_file, i_file, i_col):
    # Load and clean data
    def load(f):
        if not os.path.exists(f):
            return None
        df = pd.read_csv(f)
        df['clip_id'] = df['clip_id'].astype(str).str.replace('.mp4', '', case=False).str.strip()
        return df

    p_df = load(p_file)
    e_df = load(e_file)
    i_df = load(i_file)
    
    if p_df is None or e_df is None or i_df is None:
        print(f"\nSkipping {label}: Required files not found.")
        return

    # Merge all three sources
    m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
    m = pd.merge(m, i_df[['clip_id', i_col]], on='clip_id')
    m.columns = ['id', 'p', 'e', 'i']

    # Threshold match logic (abs diff <= 1)
    def is_match(a, b): return abs(a - b) <= 1

    results = {
        "Q1 (Case 1.1): p=e=i": 0,
        "Q2: p=i (but p!=e) Total": 0,
        "  - Case 2.1: e matches neither p nor i": 0,
        "  - Case 2.2: e matches i (but not p)": 0,
        "Q3: p=e (but p!=i) Total": 0,
        "  - Case 3.1: i matches neither p nor e": 0,
        "  - Case 3.2: i matches e (but not p)": 0,
        "No Match (Other permutations)": 0
    }

    for _, row in m.iterrows():
        pe = is_match(row['p'], row['e'])
        pi = is_match(row['p'], row['i'])
        ei = is_match(row['e'], row['i'])

        if pe and pi: 
            results["Q1 (Case 1.1): p=e=i"] += 1
        elif pi and not pe: 
            results["Q2: p=i (but p!=e) Total"] += 1
            if not ei:
                results["  - Case 2.1: e matches neither p nor i"] += 1
            else:
                results["  - Case 2.2: e matches i (but not p)"] += 1
        elif pe and not pi:
            results["Q3: p=e (but p!=i) Total"] += 1
            if not ei:
                results["  - Case 3.1: i matches neither p nor e"] += 1
            else:
                results["  - Case 3.2: i matches e (but not p)"] += 1
        else:
            results["No Match (Other permutations)"] += 1

    print(f"\n{label}")
    print("=" * 45)
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    # 1. VALENCE - SELF REPORTED
    run_analysis("1. VALENCE (Self-Reported)", 'valence', 
                 'perceived_valence_summary.csv', 'final_expressed_valence.csv', 
                 'induced_valence_summary.csv', 'avg_valence')
    
    # 2. AROUSAL - SELF REPORTED
    run_analysis("2. AROUSAL (Self-Reported)", 'arousal', 
                 'perceived_arousal_summary.csv', 'final_expressed_arousal.csv', 
                 'induced_arousal_summary.csv', 'avg_arousal')
    
    # 3. VALENCE - PHYSIOLOGICAL (BODY)
    run_analysis("3. VALENCE (Physiological)", 'valence', 
                 'perceived_valence_summary.csv', 'final_expressed_valence.csv', 
                 'inferred_valence_summary.csv', 'avg_inferred_valence')
    
    # 4. AROUSAL - PHYSIOLOGICAL (BODY)
    run_analysis("4. AROUSAL (Physiological)", 'arousal', 
                 'perceived_arousal_summary.csv', 'final_expressed_arousal.csv', 
                 'inferred_arousal_summary.csv', 'avg_inferred_arousal')