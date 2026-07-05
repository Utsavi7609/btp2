import pandas as pd

# Load all 3 model outputs
df_llama = pd.read_csv("llm_results_llama.csv")
df_mistral = pd.read_csv("llm_results_mistral.csv")
df_llama4 = pd.read_csv("llm_results_llama4.csv")

# Merge all three
df = df_llama.merge(
    df_mistral,
    on=["participant","clip_title","clip_order","timestamp"],
    suffixes=("_llama","_mistral")
)

df = df.merge(
    df_llama4,
    on=["participant","clip_title","clip_order","timestamp"],
    suffixes=("", "_llama4")
)

print("Total rows:", len(df))


# Columns to check
cols = [
    "self_reported_valence",
    "self_reported_arousal",

    "inferred_valence_llama",
    "inferred_arousal_llama",

    "inferred_valence_mistral",
    "inferred_arousal_mistral",

    "inferred_valence",
    "inferred_arousal"
]

# Rename llama4 columns properly (after merge)
df.rename(columns={
    "inferred_valence": "inferred_valence_llama4",
    "inferred_arousal": "inferred_arousal_llama4"
}, inplace=True)

cols[-2] = "inferred_valence_llama4"
cols[-1] = "inferred_arousal_llama4"


# Check missing
missing = df[cols].isna().any(axis=1)

print("Rows with ANY missing:", missing.sum())

print("\nSample problematic rows:\n")
print(df[missing].head())


# Save problematic rows
df[missing].to_csv("rows_with_missing.csv", index=False)

# Clean dataset
df_clean = df.dropna()

print("\nClean rows:", len(df_clean))

df_clean.to_csv("clean_dataset.csv", index=False)
