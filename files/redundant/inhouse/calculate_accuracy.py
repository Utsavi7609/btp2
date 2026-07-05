# import pandas as pd

# # Load the final results
# df = pd.read_csv('llm_results_final.csv')

# # 1. Define what a 'Match' is
# # We check if the AI guess is exactly the same as the user's report
# df['valence_match'] = df['self_reported_valence'] == df['inferred_valence']
# df['arousal_match'] = df['self_reported_arousal'] == df['inferred_arousal']
# df['full_match'] = df['valence_match'] & df['arousal_match']

# # 2. Calculate accuracy per participant
# summary = df.groupby('participant').agg({
#     'valence_match': 'mean',
#     'arousal_match': 'mean',
#     'full_match': 'mean'
# }) * 100 # Convert to percentage

# # 3. Overall Accuracy
# overall_accuracy = df['full_match'].mean() * 100

# print("--- PHASE 3: RELIABILITY SUMMARY ---")
# print(summary.round(2))
# print(f"\nOverall Match Rate: {overall_accuracy:.2f}%")
# print("\nLow match rates indicate that physiological signals are a better measure of emotion for those users.")