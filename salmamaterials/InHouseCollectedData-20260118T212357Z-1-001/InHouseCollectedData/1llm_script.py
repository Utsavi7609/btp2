# """
# llm_script.py
# =============
# INNER DIRECTORY: InHouseCollectedData/

# Creates LLM prompts following the Health-LLM (Kim et al., 2024) framework.
# Uses the "All" context strategy (Table 1 of Health-LLM paper):
#   - Health Context (hc):   domain knowledge about valence/arousal physiology
#   - User Context (uc):     age, gender, height, weight from user_profiles.csv
#   - Temporal Context (tc): raw HR sequence (Natural Language String method, §3.4)
#   - Statistical summary:   mean, std, min, max, drift, RMSSD

# Also follows StressLLM (Thapa et al., 2025) prompt structure:
#   - Preamble: task description + biomarker history
#   - Example Format: output specification

# Input:  fitbit_hr_context.csv
# Output: fitbit_ready_for_llm.csv
# """

# import pandas as pd
# import os

# # ─── LOAD DATA ────────────────────────────────────────────────────────────────

# fitbit_df = pd.read_csv('fitbit_hr_context.csv')
# print(f"Loaded {len(fitbit_df)} rows from fitbit_hr_context.csv")

# # Map inner-directory folder names → xlsx participant names
# FOLDER_TO_XLSX = {
#     'Nishant':  'Nishant',
#     'Debjit':   'debjit2001',
#     'pranjal':  'pranjal123',
#     'aritra':   'aritra',
#     'sayantan': 'sayantankuila',
#     'Ajay':     'ajaycc17',
# }

# # Load user profiles from outer directory
# profiles_dict = {}
# profile_path = '../user_profiles.csv'
# if os.path.exists(profile_path):
#     profiles_df = pd.read_csv(profile_path)
#     # user_id in profiles = folder name; map to xlsx participant name
#     for _, prow in profiles_df.iterrows():
#         folder = prow['user_id']
#         xlsx_name = FOLDER_TO_XLSX.get(folder, folder)
#         profiles_dict[xlsx_name] = {
#             'age':    prow.get('age', 'unknown'),
#             'gender': str(prow.get('gender', 'unknown')).lower(),
#             'height': prow.get('height', 'unknown'),
#             'weight': prow.get('weight', 'unknown'),
#         }
#     print(f"Loaded profiles for: {list(profiles_dict.keys())}")
# else:
#     print(f"⚠️  user_profiles.csv not found at {profile_path}")
#     print("   Prompts will omit demographics. Run extract_user_profiles_WINDOWS.py first.")


# # ─── PROMPT CONSTRUCTION ─────────────────────────────────────────────────────

# def create_prompt(row):
#     """
#     Health-LLM 'All' context (Table 1, Kim et al. 2024):
#     Context = C*_health + C*_user + TE*(TimeSeriesData)

#     Prompt structure:
#       [Instruction] + [Health Context] + [User Context] +
#       [Temporal Sensor Data] + [Question] + [Output Format]
#     """
#     participant = row['participant']
#     profile     = profiles_dict.get(participant, {})

#     # ── Health Context (hc) ──────────────────────────────────────────────────
#     # Injects domain knowledge about what we are predicting.
#     # Health-LLM ablation shows this gives the biggest single improvement.
#     health_ctx = (
#         "Valence refers to the subjective emotional positivity or negativity of an experience, "
#         "measured on a scale from 1 (very negative / sad) to 5 (very positive / happy). "
#         "Arousal refers to the degree of physiological and psychological activation, "
#         "measured on a scale from 1 (very calm / relaxed) to 5 (very excited / activated). "
#         "Heart rate typically increases with higher arousal states. "
#         "Heart rate variability (RMSSD) tends to decrease under negative valence and high stress. "
#         "The drift (change) in heart rate over the viewing period reflects emotional engagement."
#     )

#     # ── User Context (uc) ────────────────────────────────────────────────────
#     # StressLLM shows age adds ~8.5% improvement; gender adds ~0.4%
#     if profile:
#         user_ctx = (
#             f"The user is a {profile['age']}-year-old {profile['gender']} "
#             f"with height {profile['height']} cm and weight {profile['weight']} kg."
#         )
#     else:
#         user_ctx = "User demographic profile not available."

#     # ── Temporal Context (tc) ────────────────────────────────────────────────
#     # Natural Language String method (Health-LLM §3.4, Appendix B.0.1)
#     # "empirically observed to show the best performance"
#     hr_seq = row['hr_sequence']
#     # hr_sequence is stored as a string representation of a list in CSV
#     if isinstance(hr_seq, str):
#         hr_seq_str = hr_seq
#     else:
#         hr_seq_str = str(list(hr_seq))

#     temporal_ctx = (
#         f"[Heart Rate Sequence (BPM)]: {hr_seq_str}. "
#         f"[Statistical Summary]: "
#         f"Mean = {row['avg_bpm']} BPM, "
#         f"Std = {row['hr_std']} BPM, "
#         f"Min = {row['hr_min']} BPM, "
#         f"Max = {row['hr_max']} BPM, "
#         f"Drift (last − first) = {row['bpm_drift']} BPM, "
#         f"HRV RMSSD = {row['hrv_rmssd']} ms."
#     )

#     activity_ctx = f"The user is watching a short video clip titled '{row['clip_title']}'."

#     # ── Full Prompt ───────────────────────────────────────────────────────────
#     prompt = (
#         f"### Instruction: You are an intelligent healthcare agent specializing in "
#         f"affective computing and physiological signal analysis.\n\n"

#         f"### Health Knowledge: {health_ctx}\n\n"

#         f"### User Profile: {user_ctx}\n\n"

#         f"### Activity: {activity_ctx}\n\n"

#         f"### Sensor Readings: {temporal_ctx}\n\n"

#         f"### Question: Based on the physiological data above, predict the INDUCED "
#         f"Valence and Arousal scores that the user experienced while watching the clip. "
#         f"Use a 1–5 integer scale.\n\n"

#         f"### Response: Return ONLY a JSON object with no explanation: "
#         f'{{\"valence\": X, \"arousal\": Y}}'
#     )

#     return prompt.strip()


# # ─── CREATE PROMPTS ───────────────────────────────────────────────────────────

# fitbit_df['llm_question'] = fitbit_df.apply(create_prompt, axis=1)
# fitbit_df.to_csv('fitbit_ready_for_llm.csv', index=False)

# print(f"✅ Created {len(fitbit_df)} prompts → fitbit_ready_for_llm.csv")
# print(f"\n── Sample prompt (first row) ──────────────────────────────────────────")
# print(fitbit_df['llm_question'].iloc[0])



"""
llm_script.py
=============
INNER DIRECTORY: InHouseCollectedData/

Creates LLM prompts following the Health-LLM (Kim et al., 2024) framework.
Uses the "All" context strategy (Table 1 of Health-LLM paper):
  - Health Context (hc):   domain knowledge about valence/arousal physiology
  - User Context (uc):     age, gender, height, weight from user_profiles.csv
  - Temporal Context (tc): raw HR sequence (Natural Language String method, §3.4)
  - Statistical summary:   mean, std, min, max, drift, RMSSD

Also follows StressLLM (Thapa et al., 2025) prompt structure:
  - Preamble: task description + biomarker history
  - Example Format: output specification

Input:  fitbit_hr_context.csv
Output: fitbit_ready_for_llm.csv
"""

import pandas as pd
import os

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

fitbit_df = pd.read_csv('fitbit_hr_context.csv')
print(f"Loaded {len(fitbit_df)} rows from fitbit_hr_context.csv")

# Map inner-directory folder names → xlsx participant names
FOLDER_TO_XLSX = {
    'Nishant':  'Nishant',
    'Debjit':   'debjit2001',
    'pranjal':  'pranjal123',
    'aritra':   'aritra',
    'sayantan': 'sayantankuila',
    'Ajay':     'ajaycc17',
}

# Load user profiles from outer directory
profiles_dict = {}
profile_path = '../user_profiles.csv'
if os.path.exists(profile_path):
    profiles_df = pd.read_csv(profile_path)
    # user_id in profiles = folder name; map to xlsx participant name
    for _, prow in profiles_df.iterrows():
        folder = prow['user_id']
        xlsx_name = FOLDER_TO_XLSX.get(folder, folder)
        profiles_dict[xlsx_name] = {
            'age':    prow.get('age', 'unknown'),
            'gender': str(prow.get('gender', 'unknown')).lower(),
            'height': prow.get('height', 'unknown'),
            'weight': prow.get('weight', 'unknown'),
        }
    print(f"Loaded profiles for: {list(profiles_dict.keys())}")
else:
    print(f"⚠️  user_profiles.csv not found at {profile_path}")
    print("   Prompts will omit demographics. Run extract_user_profiles_WINDOWS.py first.")


# ─── PROMPT CONSTRUCTION ─────────────────────────────────────────────────────

def create_prompt(row):
    """
    Health-LLM 'All' context (Table 1, Kim et al. 2024):
    Context = C*_health + C*_user + TE*(TimeSeriesData)

    Prompt structure:
      [Instruction] + [Health Context] + [User Context] +
      [Temporal Sensor Data] + [Question] + [Output Format]
    """
    participant = row['participant']
    profile     = profiles_dict.get(participant, {})

    # ── Health Context (hc) ──────────────────────────────────────────────────
    health_ctx = (
        "Valence represents emotional positivity or negativity on a 1–5 scale: "
        "1 = very negative (sad, unpleasant), "
        "2 = negative, "
        "3 = neutral, "
        "4 = positive, "
        "5 = very positive (happy, pleasant). "

        "Arousal represents physiological activation on a 1–5 scale: "
        "1 = very calm / relaxed, "
        "2 = calm, "
        "3 = moderate activation, "
        "4 = excited / alert, "
        "5 = very excited / highly activated. "

        "Heart rate typically increases with higher arousal states. "
        "Heart rate variability (RMSSD) tends to decrease under stress or negative emotional states. "
        "Heart rate drift across the viewing window reflects emotional engagement."
    )

    # ── User Context (uc) ────────────────────────────────────────────────────
    if profile:
        user_ctx = (
            f"The user is a {profile['age']}-year-old {profile['gender']} "
            f"with height {profile['height']} cm and weight {profile['weight']} kg."
        )
    else:
        user_ctx = "User demographic profile not available."

    # ── Temporal Context (tc) ────────────────────────────────────────────────
    hr_seq = row['hr_sequence']
    if isinstance(hr_seq, str):
        hr_seq_str = hr_seq
    else:
        hr_seq_str = str(list(hr_seq))

    temporal_ctx = (
        f"[Heart Rate Sequence (BPM)]: {hr_seq_str}. "
        f"[Statistical Summary]: "
        f"Mean = {row['avg_bpm']} BPM, "
        f"Std = {row['hr_std']} BPM, "
        f"Min = {row['hr_min']} BPM, "
        f"Max = {row['hr_max']} BPM, "
        f"Drift (last − first) = {row['bpm_drift']} BPM, "
        f"HRV RMSSD = {row['hrv_rmssd']} ms."
    )

    activity_ctx = f"The user is watching a short video clip titled '{row['clip_title']}'."

    # ── Full Prompt ───────────────────────────────────────────────────────────
    prompt = (
        f"### Instruction: You are an intelligent healthcare agent specializing in "
        f"affective computing and physiological signal analysis.\n\n"

        f"### Health Knowledge: {health_ctx}\n\n"

        f"### User Profile: {user_ctx}\n\n"

        f"### Activity: {activity_ctx}\n\n"

        f"### Sensor Readings: {temporal_ctx}\n\n"

        f"### Question: Based on the physiological data above, predict the INDUCED "
        f"Valence and Arousal scores that the user experienced while watching the clip. "
        f"Use a 1–5 integer scale.\n\n"

        f"### Response: Return ONLY a JSON object with no explanation: "
        f'{{\"valence\": X, \"arousal\": Y}}'
    )

    return prompt.strip()


# ─── CREATE PROMPTS ───────────────────────────────────────────────────────────

fitbit_df['llm_question'] = fitbit_df.apply(create_prompt, axis=1)
fitbit_df.to_csv('fitbit_ready_for_llm.csv', index=False)

print(f"✅ Created {len(fitbit_df)} prompts → fitbit_ready_for_llm.csv")
print(f"\n── Sample prompt (first row) ──────────────────────────────────────────")
print(fitbit_df['llm_question'].iloc[0])