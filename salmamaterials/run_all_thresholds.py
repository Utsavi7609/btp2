# # import pandas as pd
# # import os

# # # This script automates the threshold testing for 0.25, 0.5, 0.75, and 1.0
# # def clean_id(series):
# #     return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# # def run_threshold_analysis(threshold):
# #     results = []
# #     for dim in ['valence', 'arousal']:
# #         # Load the 4 required data sources
# #         p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
# #         e_df = pd.read_csv(f'final_expressed_{dim}.csv')
# #         irep_df = pd.read_csv(f'induced_{dim}_summary.csv')
# #         ipred_df = pd.read_csv(f'inferred_{dim}_summary.csv')

# #         # Standardize IDs
# #         for df in [p_df, e_df, irep_df, ipred_df]:
# #             df['clip_id'] = clean_id(df['clip_id'])

# #         # Join everything into one Master Table
# #         m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
# #         m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id')
# #         m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
# #         m.columns = ['id', 'p', 'e', 'i_self', 'i_body']

# #         # Logic for Matching
# #         def is_m(a, b): return abs(a - b) <= threshold

# #         # Count Categories for SELF-REPORTED
# #         self_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_self), axis=1)])
# #         self_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_self) and not is_m(r.p, r.e), axis=1)])
# #         self_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_self), axis=1)])
# #         self_none = 192 - (self_match + self_feel + self_drift)

# #         # Count Categories for BODY (PHYSIOLOGICAL)
# #         body_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_body), axis=1)])
# #         body_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_body) and not is_m(r.p, r.e), axis=1)])
# #         body_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_body), axis=1)])
# #         body_none = 192 - (body_match + body_feel + body_drift)

# #         results.append({
# #             'Dimension': dim.capitalize(),
# #             'Threshold': threshold,
# #             'Self_Match': self_match,
# #             'Body_Match': body_match,
# #             'Self_Feel': self_feel,
# #             'Body_Feel': body_feel,
# #             'Self_Drift': self_drift,
# #             'Body_Drift': body_drift,
# #             'Self_None': self_none,
# #             'Body_None': body_none
# #         })
# #     return results

# # # --- EXECUTION ---
# # all_final_data = []
# # thresholds = [0.25, 0.5, 0.75, 1.0]

# # for t in thresholds:
# #     print(f"Processing Threshold: {t}...")
# #     all_final_data.extend(run_threshold_analysis(t))

# # # Create the final Master Table
# # df_final = pd.DataFrame(all_final_data)

# # # Print a clean version of the table to the console
# # for t in thresholds:
# #     print(f"\n--- TABLE FOR THRESHOLD {t} ---")
# #     sub = df_final[df_final['Threshold'] == t]
# #     # Formatting to match the image you provided
# #     print(f"{'Category':<30} | {'Self (V)':<8} | {'Body (V)':<8} | {'Self (A)':<8} | {'Body (A)':<8}")
# #     print("-" * 75)
# #     print(f"{'All Match (P=E=I)':<30} | {sub.iloc[0]['Self_Match']:<8} | {sub.iloc[0]['Body_Match']:<8} | {sub.iloc[1]['Self_Match']:<8} | {sub.iloc[1]['Body_Match']:<8}")
# #     print(f"{'Feel Perception (P=I, P!=E)':<30} | {sub.iloc[0]['Self_Feel']:<8} | {sub.iloc[0]['Body_Feel']:<8} | {sub.iloc[1]['Self_Feel']:<8} | {sub.iloc[1]['Body_Feel']:<8}")
# #     print(f"{'Understand/Drift (P=E, P!=I)':<30} | {sub.iloc[0]['Self_Drift']:<8} | {sub.iloc[0]['Body_Drift']:<8} | {sub.iloc[1]['Self_Drift']:<8} | {sub.iloc[1]['Body_Drift']:<8}")
# #     print(f"{'No Pair Match':<30} | {sub.iloc[0]['Self_None']:<8} | {sub.iloc[0]['Body_None']:<8} | {sub.iloc[1]['Self_None']:<8} | {sub.iloc[1]['Body_None']:<8}")

# # df_final.to_csv('MASTER_THRESHOLD_RESULTS.csv', index=False)
# # print("\nSUCCESS: 'MASTER_THRESHOLD_RESULTS.csv' created in your folder.")



# import pandas as pd
# import os

# # This script automates the threshold testing for 0.25, 0.5, 0.75, and 1.0
# def clean_id(series):
#     return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

# def run_threshold_analysis(threshold):
#     results = []
#     for dim in ['valence', 'arousal']:
#         # Load the 4 required data sources
#         p_df = pd.read_csv(f'perceived_{dim}_summary.csv')
#         e_df = pd.read_csv(f'final_expressed_{dim}.csv')
#         irep_df = pd.read_csv(f'induced_{dim}_summary.csv')
#         ipred_df = pd.read_csv(f'inferred_{dim}_summary.csv')

#         # Standardize IDs
#         for df in [p_df, e_df, irep_df, ipred_df]:
#             df['clip_id'] = clean_id(df['clip_id'])

#         # Join everything into one Master Table
#         m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df[['clip_id', 'expressed_value']], on='clip_id')
#         m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id')
#         m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
#         m.columns = ['id', 'p', 'e', 'i_self', 'i_body']

#         # Logic for Matching
#         def is_m(a, b): return abs(a - b) <= threshold

#         # Count Categories for SELF
#         self_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_self), axis=1)])
#         self_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_self) and not is_m(r.p, r.e), axis=1)])
#         self_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_self), axis=1)])
#         self_none = 192 - (self_match + self_feel + self_drift)

#         # Count Categories for BODY
#         body_match = len(m[m.apply(lambda r: is_m(r.p, r.e) and is_m(r.p, r.i_body), axis=1)])
#         body_feel = len(m[m.apply(lambda r: is_m(r.p, r.i_body) and not is_m(r.p, r.e), axis=1)])
#         body_drift = len(m[m.apply(lambda r: is_m(r.p, r.e) and not is_m(r.p, r.i_body), axis=1)])
#         body_none = 192 - (body_match + body_feel + body_drift)

#         results.append({
#             'Dimension': dim.capitalize(),
#             'Threshold': threshold,
#             'Self_Match': self_match,
#             'Body_Match': body_match,
#             'Self_Feel': self_feel,
#             'Body_Feel': body_feel,
#             'Self_Drift': self_drift,
#             'Body_Drift': body_drift,
#             'Self_None': self_none,
#             'Body_None': body_none
#         })
#     return results


# # --- EXECUTION ---
# all_final_data = []
# thresholds = [0.25, 0.5, 0.75, 1.0]

# for t in thresholds:
#     print(f"Processing Threshold: {t}...")
#     all_final_data.extend(run_threshold_analysis(t))

# df_final = pd.DataFrame(all_final_data)

# # ===============================
# # 1️⃣ SELF ONLY TABLES
# # ===============================
# print("\n\n========== SELF DATA ONLY ==========")

# for t in thresholds:
#     print(f"\n--- SELF TABLE FOR THRESHOLD {t} ---")
#     sub = df_final[df_final['Threshold'] == t]

#     print(f"{'Category':<30} | {'Valence':<8} | {'Arousal':<8}")
#     print("-" * 55)

#     print(f"{'All Match (P=E=I)':<30} | {sub.iloc[0]['Self_Match']:<8} | {sub.iloc[1]['Self_Match']:<8}")
#     print(f"{'Feel Perception (P=I, P!=E)':<30} | {sub.iloc[0]['Self_Feel']:<8} | {sub.iloc[1]['Self_Feel']:<8}")
#     print(f"{'Understand/Drift (P=E, P!=I)':<30} | {sub.iloc[0]['Self_Drift']:<8} | {sub.iloc[1]['Self_Drift']:<8}")
#     print(f"{'No Pair Match':<30} | {sub.iloc[0]['Self_None']:<8} | {sub.iloc[1]['Self_None']:<8}")


# # ===============================
# # 2️⃣ BODY ONLY TABLES
# # ===============================
# print("\n\n========== BODY DATA ONLY ==========")

# for t in thresholds:
#     print(f"\n--- BODY TABLE FOR THRESHOLD {t} ---")
#     sub = df_final[df_final['Threshold'] == t]

#     print(f"{'Category':<30} | {'Valence':<8} | {'Arousal':<8}")
#     print("-" * 55)

#     print(f"{'All Match (P=E=I)':<30} | {sub.iloc[0]['Body_Match']:<8} | {sub.iloc[1]['Body_Match']:<8}")
#     print(f"{'Feel Perception (P=I, P!=E)':<30} | {sub.iloc[0]['Body_Feel']:<8} | {sub.iloc[1]['Body_Feel']:<8}")
#     print(f"{'Understand/Drift (P=E, P!=I)':<30} | {sub.iloc[0]['Body_Drift']:<8} | {sub.iloc[1]['Body_Drift']:<8}")
#     print(f"{'No Pair Match':<30} | {sub.iloc[0]['Body_None']:<8} | {sub.iloc[1]['Body_None']:<8}")


# # ===============================
# # 3️⃣ COMBINED TABLES (original)
# # ===============================
# print("\n\n========== SELF + BODY TABLES ==========")

# for t in thresholds:
#     print(f"\n--- TABLE FOR THRESHOLD {t} ---")
#     sub = df_final[df_final['Threshold'] == t]

#     print(f"{'Category':<30} | {'Self (V)':<8} | {'Body (V)':<8} | {'Self (A)':<8} | {'Body (A)':<8}")
#     print("-" * 75)

#     print(f"{'All Match (P=E=I)':<30} | {sub.iloc[0]['Self_Match']:<8} | {sub.iloc[0]['Body_Match']:<8} | {sub.iloc[1]['Self_Match']:<8} | {sub.iloc[1]['Body_Match']:<8}")
#     print(f"{'Feel Perception (P=I, P!=E)':<30} | {sub.iloc[0]['Self_Feel']:<8} | {sub.iloc[0]['Body_Feel']:<8} | {sub.iloc[1]['Self_Feel']:<8} | {sub.iloc[1]['Body_Feel']:<8}")
#     print(f"{'Understand/Drift (P=E, P!=I)':<30} | {sub.iloc[0]['Self_Drift']:<8} | {sub.iloc[0]['Body_Drift']:<8} | {sub.iloc[1]['Self_Drift']:<8} | {sub.iloc[1]['Body_Drift']:<8}")
#     print(f"{'No Pair Match':<30} | {sub.iloc[0]['Self_None']:<8} | {sub.iloc[0]['Body_None']:<8} | {sub.iloc[1]['Self_None']:<8} | {sub.iloc[1]['Body_None']:<8}")


# # Save CSV
# df_final.to_csv('MASTER_THRESHOLD_RESULTS.csv', index=False)

# print("\nSUCCESS: 'MASTER_THRESHOLD_RESULTS.csv' created in your folder.")


import pandas as pd

def clean_id(series):
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()

def run_threshold_analysis(threshold, modality):
    e_df_raw = pd.read_csv(f'final_expressed_valence_{modality}.csv')
    e_df_raw['clip_id'] = clean_id(e_df_raw['clip_id'])

    results = []
    for dim in ['valence', 'arousal']:
        p_df     = pd.read_csv(f'perceived_{dim}_summary.csv')
        irep_df  = pd.read_csv(f'induced_{dim}_summary.csv')
        ipred_df = pd.read_csv(f'inferred_{dim}_summary.csv')

        for df in [p_df, irep_df, ipred_df]:
            df['clip_id'] = clean_id(df['clip_id'])

        e_df = e_df_raw[['clip_id', f'expressed_{dim}']].rename(
            columns={f'expressed_{dim}': 'expressed_value'}
        )

        m = pd.merge(p_df[['clip_id', f'avg_{dim}']], e_df, on='clip_id')
        m = pd.merge(m, irep_df[['clip_id', f'avg_{dim}']], on='clip_id', suffixes=('', '_irep'))
        m = pd.merge(m, ipred_df[['clip_id', f'avg_inferred_{dim}']], on='clip_id')
        m.columns = ['id', 'p', 'e', 'i_self', 'i_body']

        total = len(m)  # FIX: use actual merge count, NOT hardcoded 192

        def is_m(a, b): return abs(a - b) <= threshold

        # --- SELF: store clip_id sets per category ---
        self_match_ids = set(m[m.apply(lambda r: is_m(r.p, r.e) and      is_m(r.p, r.i_self), axis=1)]['id'])
        self_feel_ids  = set(m[m.apply(lambda r: is_m(r.p, r.i_self) and not is_m(r.p, r.e), axis=1)]['id'])
        self_drift_ids = set(m[m.apply(lambda r: is_m(r.p, r.e) and  not is_m(r.p, r.i_self), axis=1)]['id'])
        self_none_ids  = set(m['id']) - self_match_ids - self_feel_ids - self_drift_ids

        # --- BODY: store clip_id sets per category ---
        body_match_ids = set(m[m.apply(lambda r: is_m(r.p, r.e) and      is_m(r.p, r.i_body), axis=1)]['id'])
        body_feel_ids  = set(m[m.apply(lambda r: is_m(r.p, r.i_body) and not is_m(r.p, r.e), axis=1)]['id'])
        body_drift_ids = set(m[m.apply(lambda r: is_m(r.p, r.e) and  not is_m(r.p, r.i_body), axis=1)]['id'])
        body_none_ids  = set(m['id']) - body_match_ids - body_feel_ids - body_drift_ids

        results.append({
            'Modality':  modality,
            'Dimension': dim.capitalize(),
            'Threshold': threshold,
            'Total':     total,
            # counts
            'Self_Match': len(self_match_ids), 'Body_Match': len(body_match_ids),
            'Self_Feel':  len(self_feel_ids),  'Body_Feel':  len(body_feel_ids),
            'Self_Drift': len(self_drift_ids), 'Body_Drift': len(body_drift_ids),
            'Self_None':  len(self_none_ids),  'Body_None':  len(body_none_ids),
            # id sets for overlap
            'Self_Match_ids': self_match_ids,  'Body_Match_ids': body_match_ids,
            'Self_Feel_ids':  self_feel_ids,   'Body_Feel_ids':  body_feel_ids,
            'Self_Drift_ids': self_drift_ids,  'Body_Drift_ids': body_drift_ids,
            'Self_None_ids':  self_none_ids,   'Body_None_ids':  body_none_ids,
        })
    return results


# --- EXECUTION ---
thresholds = [0.25, 0.5, 0.75, 1.0]
modalities = ['AUDIO', 'VIDEO']

all_data = []
for t in thresholds:
    for mod in modalities:
        print(f"Processing Threshold={t}, Modality={mod}...")
        all_data.extend(run_threshold_analysis(t, mod))

df_full = pd.DataFrame(all_data)

CATS = {
    'Match': 'All Match (P=E=I)',
    'Feel':  'Feel Perception (P=I, P!=E)',
    'Drift': 'Understand/Drift (P=E, P!=I)',
    'None':  'No Pair Match',
}

def print_count_table(label, v_row, a_row, prefix):
    print(f"\n  --- {label} ---")
    print(f"  {'Category':<32} | {'Valence':<8} | {'Arousal':<8}")
    print(f"  {'-'*55}")
    for cat, display in CATS.items():
        print(f"  {display:<32} | {int(v_row[f'{prefix}_{cat}']):<8} | {int(a_row[f'{prefix}_{cat}']):<8}")

def print_overlap_table(label, v_aud, v_vid, a_aud, a_vid, prefix):
    print(f"\n  --- {label} ---")
    print(f"  {'Category':<32} | {'Valence':<8} | {'Arousal':<8}")
    print(f"  {'-'*55}")
    for cat, display in CATS.items():
        v_ov = len(v_aud[f'{prefix}_{cat}_ids'] & v_vid[f'{prefix}_{cat}_ids'])
        a_ov = len(a_aud[f'{prefix}_{cat}_ids'] & a_vid[f'{prefix}_{cat}_ids'])
        print(f"  {display:<32} | {v_ov:<8} | {a_ov:<8}")

def get(t, mod, dim):
    return df_full[(df_full['Threshold']==t) & (df_full['Modality']==mod) & (df_full['Dimension']==dim)].iloc[0]


# ===============================
# SELF TABLES  (AUDIO + VIDEO + OVERLAP)
# ===============================
print("\n\n========== SELF DATA ==========")
for t in thresholds:
    print(f"\n{'='*65}\n  THRESHOLD {t}\n{'='*65}")
    for mod in modalities:
        print_count_table(f"SELF | {mod}", get(t, mod, 'Valence'), get(t, mod, 'Arousal'), 'Self')
    print_overlap_table(f"SELF | AUDIO ∩ VIDEO",
                        get(t,'AUDIO','Valence'), get(t,'VIDEO','Valence'),
                        get(t,'AUDIO','Arousal'), get(t,'VIDEO','Arousal'), 'Self')


# ===============================
# BODY TABLES  (AUDIO + VIDEO + OVERLAP)
# ===============================
print("\n\n========== BODY DATA ==========")
for t in thresholds:
    print(f"\n{'='*65}\n  THRESHOLD {t}\n{'='*65}")
    for mod in modalities:
        print_count_table(f"BODY | {mod}", get(t, mod, 'Valence'), get(t, mod, 'Arousal'), 'Body')
    print_overlap_table(f"BODY | AUDIO ∩ VIDEO",
                        get(t,'AUDIO','Valence'), get(t,'VIDEO','Valence'),
                        get(t,'AUDIO','Arousal'), get(t,'VIDEO','Arousal'), 'Body')


# Save CSV (drop set columns)
df_save = df_full.drop(columns=[c for c in df_full.columns if c.endswith('_ids')])
df_save.to_csv('MASTER_THRESHOLD_RESULTS_AV.csv', index=False)
print("\nSUCCESS: 'MASTER_THRESHOLD_RESULTS_AV.csv' created.")