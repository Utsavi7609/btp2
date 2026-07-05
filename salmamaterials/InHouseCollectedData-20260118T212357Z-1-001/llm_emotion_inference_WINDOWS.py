# # """
# # LLM-based Emotion Inference from Physiological Data
# # WINDOWS VERSION with HARDCODED API KEY
# # Following Health-LLM Section 3.1 (Context Enhancement)
# # Uses Groq API with Llama 3.1 70B
# # """

# # import pandas as pd
# # import numpy as np
# # from groq import Groq
# # import time
# # import json
# # from tqdm import tqdm

# # # HARDCODED API KEY (as requested)
# # GROQ_API_KEY = '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'

# # # Health-LLM Prompt Template with ALL CONTEXTS
# # PROMPT_TEMPLATE = """### Instruction: You are an intelligent healthcare agent specialized in emotion recognition from physiological signals.

# # ### Health Knowledge:
# # **Valence** represents the pleasantness dimension of emotion (1=very negative/unpleasant, 5=very positive/pleasant).
# # **Arousal** represents the activation/intensity dimension (1=very calm/low energy, 5=very excited/high energy).

# # **Physiological Indicators:**
# # - High heart rate (>80 bpm at rest) typically indicates HIGH arousal
# # - Low heart rate (<65 bpm at rest) typically indicates LOW arousal
# # - Heart rate variability (HRV/RMSSD) reflects emotional state
# # - Rapid heart rate increases often accompany negative high-arousal states (fear, anger, stress)
# # - Gradual increases may accompany positive high-arousal states (excitement, joy)
# # - Steady heart rate with low variability suggests calm states (low arousal)

# # ### User Profile:
# # - Age: {age} years old
# # - Gender: {gender}
# # - Height: {height} cm
# # - Weight: {weight} kg

# # ### Question:
# # The user's physiological data for this period shows:
# # - Heart Rate Statistics: Mean={hr_mean:.1f} bpm, Min={hr_min:.1f} bpm, Max={hr_max:.1f} bpm, Variability (std)={hr_std:.1f} bpm
# # - Heart Rate Time Series (recent samples): {hr_sequence}

# # Based on this user's profile and the physiological patterns, what emotional state does this data likely reflect?

# # **Respond with ONLY two numbers separated by a comma:**
# # valence,arousal

# # Example: 3,4 (means valence=3, arousal=4)

# # ### Response:"""

# # class GroqEmotionInference:
# #     """
# #     LLM-based emotion inference using Groq API
# #     """
    
# #     def __init__(self):
# #         self.client = Groq(api_key=GROQ_API_KEY)
# #         self.model = "llama-3.1-70b-versatile"
        
# #         print(f"✅ Initialized Groq client")
# #         print(f"   Model: {self.model}")
# #         print(f"   API Key: {GROQ_API_KEY[:20]}...{GROQ_API_KEY[-5:]}")
    
# #     def format_hr_sequence(self, hr_values, max_length=20):
# #         """
# #         Format HR as natural language string
# #         """
# #         if len(hr_values) <= max_length:
# #             return ', '.join([f"{v:.1f}" for v in hr_values])
# #         else:
# #             first_10 = ', '.join([f"{v:.1f}" for v in hr_values[:10]])
# #             last_10 = ', '.join([f"{v:.1f}" for v in hr_values[-10:]])
# #             return f"{first_10}, ..., {last_10}"
    
# #     def create_prompt(self, window_data, user_profile):
# #         """
# #         Create prompt with ALL contexts
# #         """
# #         # Parse HR values from string
# #         hr_values_str = window_data['hr_values']
# #         hr_values = [float(x) for x in hr_values_str.split(',') if x.strip()]
        
# #         hr_sequence = self.format_hr_sequence(hr_values)
        
# #         prompt = PROMPT_TEMPLATE.format(
# #             age=user_profile.get('age', 25),
# #             gender=user_profile.get('gender', 'unknown'),
# #             height=user_profile.get('height', 170),
# #             weight=user_profile.get('weight', 70),
# #             hr_mean=window_data['hr_mean'],
# #             hr_min=window_data['hr_min'],
# #             hr_max=window_data['hr_max'],
# #             hr_std=window_data['hr_std'],
# #             hr_sequence=hr_sequence
# #         )
        
# #         return prompt
    
# #     def call_llm(self, prompt, max_retries=3):
# #         """
# #         Call Groq API with retry logic
# #         """
# #         for attempt in range(max_retries):
# #             try:
# #                 response = self.client.chat.completions.create(
# #                     model=self.model,
# #                     messages=[{"role": "user", "content": prompt}],
# #                     temperature=0.0,  # Deterministic
# #                     max_tokens=50
# #                 )
                
# #                 response_text = response.choices[0].message.content.strip()
                
# #                 # Parse "valence, arousal" format
# #                 try:
# #                     response_text = response_text.replace(' ', '').replace('\n', '')
# #                     parts = response_text.split(',')
                    
# #                     if len(parts) >= 2:
# #                         valence = float(parts[0])
# #                         arousal = float(parts[1])
                        
# #                         # Clamp to [1, 5] range
# #                         valence = max(1.0, min(5.0, valence))
# #                         arousal = max(1.0, min(5.0, arousal))
                        
# #                         return {'valence': valence, 'arousal': arousal, 'raw_response': response_text}
# #                     else:
# #                         print(f"  ⚠️  Parse error: {response_text}")
# #                         return {'valence': 3.0, 'arousal': 3.0, 'raw_response': response_text}
# #                 except ValueError:
# #                     print(f"  ⚠️  Value error in: {response_text}")
# #                     return {'valence': 3.0, 'arousal': 3.0, 'raw_response': response_text}
                    
# #             except Exception as e:
# #                 print(f"  ⚠️  API error (attempt {attempt+1}): {e}")
# #                 if "rate" in str(e).lower():
# #                     wait_time = 2 ** attempt
# #                     print(f"    Waiting {wait_time}s...")
# #                     time.sleep(wait_time)
# #                 else:
# #                     break
        
# #         return {'valence': 3.0, 'arousal': 3.0, 'raw_response': 'ERROR'}
    
# #     def run_inference(self):
# #         """
# #         Run LLM inference on all windows
# #         """
# #         print("\n" + "="*60)
# #         print("LLM-based Emotion Inference")
# #         print("Following Health-LLM Methodology")
# #         print("="*60)
        
# #         # Load data
# #         print(f"\n📂 Loading data...")
        
# #         try:
# #             windows_df = pd.read_csv('fitbit_hr_windows.csv')
# #             profiles_df = pd.read_csv('user_profiles.csv')
# #         except FileNotFoundError as e:
# #             print(f"❌ Error: {e}")
# #             print("\n⚠️  Please run these scripts first:")
# #             print("   1. python extract_user_profiles_WINDOWS.py")
# #             print("   2. python extract_heart_rate_data_WINDOWS.py")
# #             return None
        
# #         print(f"  ✅ Loaded {len(windows_df)} windows")
# #         print(f"  ✅ Loaded {len(profiles_df)} user profiles")
        
# #         # Resume logic
# #         output_file = 'inferred_emotions_llm.csv'
# #         start_idx = 0
        
# #         if Path(output_file).exists():
# #             existing_df = pd.read_csv(output_file)
# #             start_idx = len(existing_df)
# #             print(f"\n♻️  Resuming from row {start_idx}...")
# #             results_df = existing_df
# #         else:
# #             results_df = pd.DataFrame()
        
# #         # Process each window
# #         print(f"\n🤖 Running LLM inference on {len(windows_df) - start_idx} windows...")
        
# #         for idx in tqdm(range(start_idx, len(windows_df)), desc="Processing"):
# #             window_row = windows_df.iloc[idx]
# #             user_id = window_row['user_id']
            
# #             # Get user profile
# #             user_profile = profiles_df[profiles_df['user_id'] == user_id]
            
# #             if user_profile.empty:
# #                 user_profile = pd.DataFrame([{
# #                     'user_id': user_id,
# #                     'age': 25,
# #                     'gender': 'unknown',
# #                     'height': 170,
# #                     'weight': 70
# #                 }])
            
# #             user_profile = user_profile.iloc[0]
            
# #             # Create prompt and call LLM
# #             prompt = self.create_prompt(window_row, user_profile)
# #             result = self.call_llm(prompt)
            
# #             # Store result
# #             result_row = {
# #                 'user_id': user_id,
# #                 'date': window_row['date'],
# #                 'hr_mean': window_row['hr_mean'],
# #                 'hr_std': window_row['hr_std'],
# #                 'hr_min': window_row['hr_min'],
# #                 'hr_max': window_row['hr_max'],
# #                 'inferred_valence': result['valence'],
# #                 'inferred_arousal': result['arousal'],
# #                 'llm_response': result['raw_response']
# #             }
            
# #             results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
            
# #             # Save checkpoint every 10 rows
# #             if (idx + 1) % 10 == 0:
# #                 results_df.to_csv(output_file, index=False)
            
# #             # Rate limiting (Groq: 30 req/min)
# #             time.sleep(2.1)
        
# #         # Final save
# #         results_df.to_csv(output_file, index=False)
        
# #         print(f"\n✅ Inference complete!")
# #         print(f"📊 Saved {len(results_df)} results to: {output_file}")
        
# #         # Statistics
# #         print(f"\n📈 Emotion Statistics:")
# #         print(f"  Valence: Mean={results_df['inferred_valence'].mean():.2f}, "
# #               f"Std={results_df['inferred_valence'].std():.2f}")
# #         print(f"  Arousal: Mean={results_df['inferred_arousal'].mean():.2f}, "
# #               f"Std={results_df['inferred_arousal'].std():.2f}")
        
# #         # Create summary files
# #         self.create_summary_files(results_df)
        
# #         return results_df
    
# #     def create_summary_files(self, df):
# #         """
# #         Create valence and arousal summary files
# #         """
# #         # Valence summary
# #         valence_df = df.pivot_table(
# #             index='date',
# #             columns='user_id',
# #             values='inferred_valence',
# #             aggfunc='mean'
# #         )
# #         valence_df['avg_inferred_valence'] = valence_df.mean(axis=1)
# #         valence_df.to_csv('inferred_valence_summary.csv')
        
# #         # Arousal summary
# #         arousal_df = df.pivot_table(
# #             index='date',
# #             columns='user_id',
# #             values='inferred_arousal',
# #             aggfunc='mean'
# #         )
# #         arousal_df['avg_inferred_arousal'] = arousal_df.mean(axis=1)
# #         arousal_df.to_csv('inferred_arousal_summary.csv')
        
# #         print(f"\n✅ Created summary files:")
# #         print(f"  📄 inferred_valence_summary.csv")
# #         print(f"  📄 inferred_arousal_summary.csv")

# # if __name__ == "__main__":
# #     from pathlib import Path
    
# #     pipeline = GroqEmotionInference()
# #     results = pipeline.run_inference()
    
# #     if results is not None:
# #         print("\n✅ Pipeline complete!")
# #         print("\n📁 Output files:")
# #         print("  1. inferred_emotions_llm.csv - Main results")
# #         print("  2. inferred_valence_summary.csv - Valence by user/date")
# #         print("  3. inferred_arousal_summary.csv - Arousal by user/date")


# """
# LLM-based Emotion Inference from Physiological Data
# WINDOWS VERSION with HARDCODED API KEY
# Following Health-LLM Section 3.1 (Context Enhancement)
# Uses Groq API with Llama 3.1 70B
# """

# import pandas as pd
# import numpy as np
# from groq import Groq
# import time
# from tqdm import tqdm
# from pathlib import Path


# GROQ_API_KEY = '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'


# PROMPT_TEMPLATE = """### Instruction
# You are an intelligent healthcare agent specialized in emotion recognition from physiological signals.

# ### Emotion Dimensions
# Valence: emotional pleasantness (1 = very negative, 5 = very positive)  
# Arousal: emotional intensity or activation (1 = very calm, 5 = very excited)

# ### Physiological Knowledge
# - Higher heart rate generally indicates higher arousal.
# - Lower heart rate generally indicates lower arousal.
# - Heart Rate Variability (HRV) reflects autonomic nervous system activity.

# HRV measured using **RMSSD (Root Mean Square of Successive Differences)**:
# - Higher RMSSD → relaxed, calm states (parasympathetic dominance)
# - Lower RMSSD → stress, anxiety, high arousal (sympathetic activation)

# ### User Profile
# Age: {age} years  
# Gender: {gender}  
# Height: {height} cm  
# Weight: {weight} kg  

# ### Physiological Measurements (2-hour window)
# Mean Heart Rate: {hr_mean:.1f} bpm  
# Minimum Heart Rate: {hr_min:.1f} bpm  
# Maximum Heart Rate: {hr_max:.1f} bpm  
# Heart Rate Standard Deviation: {hr_std:.1f} bpm  
# HRV (RMSSD): {hrv_rmssd:.2f}

# Based on the physiological measurements and user profile, estimate the user's emotional state.

# Respond with ONLY two numbers separated by a comma:

# <valence>,<arousal>

# Example:
# 3,4

# Do NOT include words or explanations.

# ### Response
# """


# class GroqEmotionInference:

#     def __init__(self):

#         self.client = Groq(api_key=GROQ_API_KEY)
#         self.model = "llama-3.3-70b-versatile"

#         print("✅ Initialized Groq client")
#         print(f"   Model: {self.model}")
#         print(f"   API Key: {GROQ_API_KEY[:20]}...{GROQ_API_KEY[-5:]}")


#     def create_prompt(self, window_data, user_profile):

#         prompt = PROMPT_TEMPLATE.format(
#             age=user_profile.get('age', 25),
#             gender=user_profile.get('gender', 'unknown'),
#             height=user_profile.get('height', 170),
#             weight=user_profile.get('weight', 70),
#             hr_mean=window_data['hr_mean'],
#             hr_min=window_data['hr_min'],
#             hr_max=window_data['hr_max'],
#             hr_std=window_data['hr_std'],
#             hrv_rmssd=window_data.get('hrv_rmssd', 0)
#         )

#         return prompt


#     def call_llm(self, prompt, max_retries=3):

#         for attempt in range(max_retries):

#             try:

#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[{"role": "user", "content": prompt}],
#                     temperature=0.0,
#                     max_tokens=50
#                 )

#                 response_text = response.choices[0].message.content.strip()

#                 try:

#                     response_text = response_text.replace(' ', '').replace('\n', '')
#                     parts = response_text.split(',')

#                     if len(parts) >= 2:

#                         valence = float(parts[0])
#                         arousal = float(parts[1])

#                         valence = max(1.0, min(5.0, valence))
#                         arousal = max(1.0, min(5.0, arousal))

#                         return {
#                             'valence': valence,
#                             'arousal': arousal,
#                             'raw_response': response_text
#                         }

#                 except:
#                     pass

#                 return {'valence': 3.0, 'arousal': 3.0, 'raw_response': response_text}

#             except Exception as e:

#                 print(f"⚠️ API error (attempt {attempt+1}): {e}")

#                 if "rate" in str(e).lower():
#                     wait_time = 2 ** attempt
#                     time.sleep(wait_time)
#                 else:
#                     break

#         return {'valence': 3.0, 'arousal': 3.0, 'raw_response': 'ERROR'}


#     def run_inference(self):

#         print("\n" + "="*60)
#         print("LLM-based Emotion Inference")
#         print("="*60)

#         windows_df = pd.read_csv('fitbit_hr_windows.csv')
#         profiles_df = pd.read_csv('user_profiles.csv')

#         print(f"Loaded {len(windows_df)} windows")

#         output_file = 'inferred_emotions_llm.csv'

#         start_idx = 0

#         if Path(output_file).exists():
#             existing_df = pd.read_csv(output_file)
#             start_idx = len(existing_df)
#             results_df = existing_df
#             print(f"Resuming from row {start_idx}")
#         else:
#             results_df = pd.DataFrame()

#         for idx in tqdm(range(start_idx, len(windows_df))):

#             window_row = windows_df.iloc[idx]
#             user_id = window_row['user_id']

#             user_profile = profiles_df[profiles_df['user_id'] == user_id]

#             if user_profile.empty:
#                 user_profile = pd.DataFrame([{
#                     'user_id': user_id,
#                     'age': 25,
#                     'gender': 'unknown',
#                     'height': 170,
#                     'weight': 70
#                 }])

#             user_profile = user_profile.iloc[0]

#             prompt = self.create_prompt(window_row, user_profile)

#             result = self.call_llm(prompt)

#             result_row = {
#                 'user_id': user_id,
#                 'date': window_row['date'],
#                 'hr_mean': window_row['hr_mean'],
#                 'hr_std': window_row['hr_std'],
#                 'hr_min': window_row['hr_min'],
#                 'hr_max': window_row['hr_max'],
#                 'hrv_rmssd': window_row.get('hrv_rmssd', 0),
#                 'inferred_valence': result['valence'],
#                 'inferred_arousal': result['arousal'],
#                 'llm_response': result['raw_response']
#             }

#             results_df = pd.concat(
#                 [results_df, pd.DataFrame([result_row])],
#                 ignore_index=True
#             )

#             if (idx + 1) % 10 == 0:
#                 results_df.to_csv(output_file, index=False)

#             time.sleep(2.1)

#         results_df.to_csv(output_file, index=False)

#         print("Inference complete")

#         return results_df


# if __name__ == "__main__":

#     pipeline = GroqEmotionInference()
#     pipeline.run_inference()



"""
LLM-based Emotion Inference from Physiological Data
WINDOWS VERSION with HARDCODED API KEY
Following Health-LLM Section 3.1 (Context Enhancement)
Uses Groq API with Llama 3.3 70B
"""

import pandas as pd
import numpy as np
from groq import Groq
import time
from tqdm import tqdm
from pathlib import Path


GROQ_API_KEY = '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'


PROMPT_TEMPLATE = """### Instruction
You are an intelligent healthcare agent specialized in emotion recognition from physiological signals.

### Emotion Dimensions
Valence: emotional pleasantness (1 = very negative, 5 = very positive)  
Arousal: emotional intensity or activation (1 = very calm, 5 = very excited)

### Physiological Knowledge
- Higher heart rate generally indicates higher arousal.
- Lower heart rate generally indicates lower arousal.
- Heart Rate Variability (HRV) reflects autonomic nervous system activity.

HRV measured using **RMSSD (Root Mean Square of Successive Differences)**:
- Higher RMSSD → relaxed, calm states (parasympathetic dominance)
- Lower RMSSD → stress, anxiety, high arousal (sympathetic activation)

### User Profile
Age: {age} years  
Gender: {gender}  
Height: {height} cm  
Weight: {weight} kg  

### Physiological Measurements (2-hour window)
Mean Heart Rate: {hr_mean:.1f} bpm  
Minimum Heart Rate: {hr_min:.1f} bpm  
Maximum Heart Rate: {hr_max:.1f} bpm  
Heart Rate Standard Deviation: {hr_std:.1f} bpm  
HRV (RMSSD): {hrv_rmssd:.2f}

Based on the physiological measurements and user profile, estimate the user's emotional state.

Respond with ONLY two numbers separated by a comma:

<valence>,<arousal>

Example:
3,4

Do NOT include words or explanations.

### Response
"""


class GroqEmotionInference:

    def __init__(self):

        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"

        print("✅ Initialized Groq client")
        print(f"   Model: {self.model}")
        print(f"   API Key: {GROQ_API_KEY[:20]}...{GROQ_API_KEY[-5:]}")


    def create_prompt(self, window_data, user_profile):

        prompt = PROMPT_TEMPLATE.format(
            age=user_profile.get('age', 25),
            gender=user_profile.get('gender', 'unknown'),
            height=user_profile.get('height', 170),
            weight=user_profile.get('weight', 70),
            hr_mean=window_data['hr_mean'],
            hr_min=window_data['hr_min'],
            hr_max=window_data['hr_max'],
            hr_std=window_data['hr_std'],
            hrv_rmssd=window_data.get('hrv_rmssd', 0)
        )

        return prompt


    def call_llm(self, prompt, max_retries=3):

        for attempt in range(max_retries):

            try:

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50
                )

                response_text = response.choices[0].message.content.strip()

                try:

                    response_text = response_text.replace(' ', '').replace('\n', '')
                    parts = response_text.split(',')

                    if len(parts) >= 2:

                        valence = float(parts[0])
                        arousal = float(parts[1])

                        valence = max(1.0, min(5.0, valence))
                        arousal = max(1.0, min(5.0, arousal))

                        return {
                            'valence': valence,
                            'arousal': arousal,
                            'raw_response': response_text
                        }

                except:
                    pass

                return {'valence': 3.0, 'arousal': 3.0, 'raw_response': response_text}

            except Exception as e:

                print(f"⚠️ API error (attempt {attempt+1}): {e}")

                if "rate" in str(e).lower():
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    break

        return {'valence': 3.0, 'arousal': 3.0, 'raw_response': 'ERROR'}


    def run_inference(self):

        print("\n" + "="*60)
        print("LLM-based Emotion Inference")
        print("="*60)

        windows_df = pd.read_csv('fitbit_hr_windows.csv')
        profiles_df = pd.read_csv('user_profiles.csv')

        print(f"Loaded {len(windows_df)} windows")

        output_file = 'inferred_emotions_llm.csv'

        start_idx = 0

        if Path(output_file).exists():

            existing_df = pd.read_csv(output_file)

            # ✅ FIX: remove rows where inference failed
            existing_df = existing_df[existing_df["llm_response"] != "ERROR"]

            start_idx = len(existing_df)
            results_df = existing_df

            print(f"Resuming from row {start_idx}")

        else:
            results_df = pd.DataFrame()

        for idx in tqdm(range(start_idx, len(windows_df))):

            window_row = windows_df.iloc[idx]
            user_id = window_row['user_id']

            user_profile = profiles_df[profiles_df['user_id'] == user_id]

            if user_profile.empty:
                user_profile = pd.DataFrame([{
                    'user_id': user_id,
                    'age': 25,
                    'gender': 'unknown',
                    'height': 170,
                    'weight': 70
                }])

            user_profile = user_profile.iloc[0]

            prompt = self.create_prompt(window_row, user_profile)

            result = self.call_llm(prompt)

            result_row = {
                'user_id': user_id,
                'date': window_row['date'],
                'hr_mean': window_row['hr_mean'],
                'hr_std': window_row['hr_std'],
                'hr_min': window_row['hr_min'],
                'hr_max': window_row['hr_max'],
                'hrv_rmssd': window_row.get('hrv_rmssd', 0),
                'inferred_valence': result['valence'],
                'inferred_arousal': result['arousal'],
                'llm_response': result['raw_response']
            }

            results_df = pd.concat(
                [results_df, pd.DataFrame([result_row])],
                ignore_index=True
            )

            if (idx + 1) % 10 == 0:
                results_df.to_csv(output_file, index=False)

            time.sleep(2.1)

        results_df.to_csv(output_file, index=False)

        print("Inference complete")

        return results_df


if __name__ == "__main__":

    pipeline = GroqEmotionInference()
    pipeline.run_inference()