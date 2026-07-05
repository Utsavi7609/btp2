"""
Enhanced LLM-based Emotion Inference from Physiological Data
Uses Groq API with Qwen and Llama
Enhanced with K-EmoCon Clinical Few-Shot Anchors
"""

import pandas as pd
import numpy as np
from groq import Groq
import time
from tqdm import tqdm
from pathlib import Path
import sys

import re

# STARTING KEY
GROQ_API_KEY = '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'

PROMPT_TEMPLATE = """### Instruction
You are an intelligent healthcare agent specializing in affective computing and physiological signal analysis.

### Health Knowledge
**Valence** refers to the subjective emotional positivity or negativity, measured from 1 (very negative / sad) to 5 (very positive / happy).
**Arousal** refers to physiological and psychological activation level, measured from 1 (very calm / relaxed) to 5 (very excited / activated).

**Physiological Indicators:**
- Higher heart rate (>90 bpm) typically indicates HIGH arousal (4-5)
- Lower heart rate (<70 bpm) typically indicates LOW arousal (1-2)
- Moderate heart rate (70-90 bpm) typically indicates MODERATE arousal (2-4)
- Heart Rate Variability (RMSSD) reflects autonomic nervous system activity
- Higher RMSSD (>30) suggests relaxed, calm states (positive valence, low arousal)
- Lower RMSSD (<15) suggests stress or high activation (lower valence, high arousal)
- High HR standard deviation indicates emotional volatility
- Low HR standard deviation indicates stable emotional state

### K-EmoCon Clinically Validated Few-Shot Examples
**Example 1:** Mean HR = 89.5 bpm, Std = 10.8, Min = 69.4, Max = 118.2, RMSSD = 53789.4
Result: Valence = 3.0, Arousal = 2.7

**Example 2:** Mean HR = 80.5, Std = 9.0, Min = 59.4, Max = 111.4, RMSSD = 83844.6
Result: Valence = 2.7, Arousal = 3.2

**Example 3:** Mean HR = 80.7, Std = 9.8, Min = 56.7, Max = 103.0, RMSSD = 74079.5
Result: Valence = 2.8, Arousal = 2.7

**Example 4:** Mean HR = 67.2, Std = 3.9, Min = 59.2, Max = 80.5, RMSSD = 108843.0
Result: Valence = 3.4, Arousal = 2.1

**Example 5:** Mean HR = 102.3, Std = 14.2, Min = 72.1, Max = 135.8, RMSSD = 22145.3
Result: Valence = 2.3, Arousal = 3.8

### User Profile
Age: {age} years
Gender: {gender}
Height: {height} cm
Weight: {weight} kg

### Current Physiological Measurements (2-hour window)
Mean Heart Rate: {hr_mean:.1f} bpm
Minimum Heart Rate: {hr_min:.1f} bpm
Maximum Heart Rate: {hr_max:.1f} bpm
Heart Rate Standard Deviation: {hr_std:.1f} bpm
HRV (RMSSD): {hrv_rmssd:.2f}

Based on the physiological measurements, user profile, and calibrated clinical examples above, predict this user's emotional state.

IMPORTANT: Use the few-shot examples to calibrate your predictions. Do NOT default to neutral values. Use decimal precision (e.g., 2.8, 3.5).

Respond with ONLY two decimal numbers separated by a comma:
<valence>,<arousal>

Example: 2.8,3.5
Do NOT include words or explanations.
### Response
/no_think
"""

class GroqEmotionInference:
    def __init__(self):
        global GROQ_API_KEY
        self.api_key = GROQ_API_KEY
        self.client = Groq(api_key=self.api_key)
        self.model = "qwen-2.5-32b"

    def create_prompt(self, window_data, user_profile):
        return PROMPT_TEMPLATE.format(
            age=user_profile.get('age', 25),
            gender=user_profile.get('gender', 'unknown'),
            height=user_profile.get('height', 170),
            weight=user_profile.get('weight', 70),
            hr_mean=window_data['avg_bpm'],
            hr_min=window_data['hr_min'],
            hr_max=window_data['hr_max'],
            hr_std=window_data['hr_std'],
            hrv_rmssd=window_data.get('hrv_rmssd', 0)
        )

    def call_llm(self, prompt, idx_for_logging):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=150
                )
                response_text = response.choices[0].message.content.strip()

                try:
                    # Strip <think> blocks completely in case /no_think fails
                    clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                    
                    # Extract the last two decimal numbers found in the text
                    numbers = re.findall(r'\d+\.?\d*', clean)
                    if len(numbers) >= 2:
                        valence = max(1.0, min(5.0, float(numbers[-2])))
                        arousal = max(1.0, min(5.0, float(numbers[-1])))
                        return {'valence': round(valence, 2), 'arousal': round(arousal, 2), 'raw_response': response_text}
                except:
                    pass

                return {'valence': None, 'arousal': None, 'raw_response': response_text}

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    print(f"\n🚨 GROQ RATE LIMIT REACHED AT ROW {idx_for_logging}! 🚨")
                    new_key = input("\n👉 Paste a new Groq API key here and press Enter (or press Enter to abort and pause): ").strip()
                    
                    if new_key:
                        self.api_key = new_key
                        self.client = Groq(api_key=self.api_key)
                        print("✅ Key swapped! Resuming from the exact same row...\n")
                        continue
                    else:
                        print("Aborting. You can run the script again later to resume.")
                        sys.exit(1)
                elif "decommissioned" in err_str or "404" in err_str:
                    print(f"\n❌ Model {self.model} is not supported on this account or decommissioned!")
                    print("Available Groq models usually include: llama-3.3-70b-versatile, qwen-2.5-32b")
                    sys.exit(1)
                else:
                    print(f"API error: {e}")
                    time.sleep(5)

    def run_inference(self):
        print("\n" + "="*60)
        print("Model Bake-Off: Qwen vs Llama (Hot-Swap Enabled)")
        print("="*60)

        windows_df = pd.read_csv('InHouseCollectedData/fitbit_ready_for_llm_v3_shifted.csv')
        try:
            profiles_df = pd.read_csv('user_profiles.csv')
        except FileNotFoundError:
            profiles_df = pd.DataFrame(columns=['user_id', 'age', 'gender', 'height', 'weight'])

        print(f"Loaded {len(windows_df)} clip-level windows")
        
        # Testing Llama-3.3-70B exclusively for the definitive BTP-2 pipeline
        models_to_test = {
            "llama": "llama-3.3-70b-versatile"
        }

        for model_name, model_id in models_to_test.items():
            print(f"\n🚀 Starting run for: {model_name} ({model_id})")
            self.model = model_id
            output_file = f'_inferred_clip_emotions_{model_name}.csv'

            start_idx = 0

            if Path(output_file).exists():
                try:
                    existing_df = pd.read_csv(output_file)
                    # Filter out ones that got "ERROR" or where valence is null
                    existing_df = existing_df[(existing_df["llm_response"] != "ERROR") & (existing_df["inferred_valence"].notna())]
                    start_idx = len(existing_df)
                    results_df = existing_df
                    if start_idx >= len(windows_df):
                        print(f"✅ {model_name} is already 100% complete! Skipping.")
                        continue
                    print(f"Resuming from row {start_idx}")
                except Exception:
                    # File is empty or corrupted from an abort
                    start_idx = 0
                    results_df = pd.DataFrame()
            else:
                results_df = pd.DataFrame()

            for idx in tqdm(range(start_idx, len(windows_df))):

                window_row = windows_df.iloc[idx]
                user_id = window_row['participant']

                user_profile = profiles_df[profiles_df['user_id'] == user_id]
                if user_profile.empty:
                    user_profile = pd.DataFrame([{'user_id': user_id, 'age': 25, 'gender': 'unknown', 'height': 170, 'weight': 70}])
                user_profile = user_profile.iloc[0]

                prompt = self.create_prompt(window_row, user_profile)
                result = self.call_llm(prompt, idx)

                if result['valence'] is None:
                    print(f"\n⚠️ Failed to parse response at row {idx}: {result['raw_response']}")
                    # Save what we have and abort so we don't corrupt the CSV
                    results_df.to_csv(output_file, index=False)
                    sys.exit(1)

                result_row = {
                    'participant': user_id,
                    'clip_title': window_row.get('clip_title', ''),
                    'timestamp': window_row.get('timestamp', ''),
                    'hr_mean': window_row['avg_bpm'],
                    'hr_std': window_row['hr_std'],
                    'hr_min': window_row['hr_min'],
                    'hr_max': window_row['hr_max'],
                    'hrv_rmssd': window_row.get('hrv_rmssd', 0),
                    'inferred_valence': result['valence'],
                    'inferred_arousal': result['arousal'],
                    'llm_response': result['raw_response']
                }

                results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)

                if (idx + 1) % 10 == 0:
                    results_df.to_csv(output_file, index=False)

                time.sleep(0.5)

            results_df.to_csv(output_file, index=False)

            print(f"✅ {model_name} complete! Saved to {output_file}")
            print(f"   Valence: Mean={results_df['inferred_valence'].mean():.2f}, Std={results_df['inferred_valence'].std():.2f}")
            print(f"   Arousal: Mean={results_df['inferred_arousal'].mean():.2f}, Std={results_df['inferred_arousal'].std():.2f}")

if __name__ == "__main__":
    pipeline = GroqEmotionInference()
    pipeline.run_inference()
