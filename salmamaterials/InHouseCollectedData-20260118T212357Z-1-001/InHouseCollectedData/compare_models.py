"""
compare_models.py
=================
INNER DIRECTORY: InHouseCollectedData/

Compares Llama vs Mistral predictions against self-reported (ground truth) labels.

Evaluation metrics follow Health-LLM (Kim et al., 2024) Table 3:
  - MAE  (Mean Absolute Error) ← primary metric, lower is better
  - Pearson r                  ← correlation with ground truth
  - Accuracy within ±1         ← practical tolerance metric

Also follows check_reported_vs_inferred.py approach:
  - Per-clip average comparison
  - Per-participant breakdown

Ground truth: self_reported_valence / self_reported_arousal
              (= impact_valence / impact_arousal from clip_responses_export.xlsx)

Inputs:  llm_results_llama.csv, llm_results_mistral.csv
Outputs: model_comparison_results.csv, per_clip_comparison.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


# ─── HELPER ───────────────────────────────────────────────────────────────────

def clean_id(series):
    """Strip .mp4 extension for clean clip matching."""
    return series.astype(str).str.replace('.mp4', '', case=False).str.strip()


def evaluate_model(df, model_name):
    """
    Compute MAE, Pearson r, and Accuracy (within ±1) for valence and arousal.
    Follows Health-LLM Table 3 evaluation methodology.
    """
    df = df.copy()
    df['inferred_valence'] = pd.to_numeric(df['inferred_valence'], errors='coerce')
    df['inferred_arousal'] = pd.to_numeric(df['inferred_arousal'], errors='coerce')
    df['self_reported_valence'] = pd.to_numeric(df['self_reported_valence'], errors='coerce')
    df['self_reported_arousal'] = pd.to_numeric(df['self_reported_arousal'], errors='coerce')

    valid_mask = (
        df['inferred_valence'].notna() & df['inferred_arousal'].notna() &
        df['self_reported_valence'].notna() & df['self_reported_arousal'].notna()
    )
    df_v = df[valid_mask]
    n    = len(df_v)

    if n < 3:
        print(f"\n{model_name}: Only {n} valid rows — insufficient for evaluation.")
        return {}

    pv = df_v['inferred_valence'].values.astype(float)
    pa = df_v['inferred_arousal'].values.astype(float)
    tv = df_v['self_reported_valence'].values.astype(float)
    ta = df_v['self_reported_arousal'].values.astype(float)

    mae_v = float(np.mean(np.abs(pv - tv)))
    mae_a = float(np.mean(np.abs(pa - ta)))

    r_v, p_v = pearsonr(pv, tv)
    r_a, p_a = pearsonr(pa, ta)

    # Accuracy within ±1 (same threshold as check_reported_vs_inferred.py)
    acc_v    = float(np.mean(np.abs(pv - tv) <= 1.0) * 100)
    acc_a    = float(np.mean(np.abs(pa - ta) <= 1.0) * 100)
    acc_both = float(np.mean((np.abs(pv - tv) <= 1.0) & (np.abs(pa - ta) <= 1.0)) * 100)

    # Exact match (strict)
    exact_v = float(np.mean(pv.round() == tv) * 100)
    exact_a = float(np.mean(pa.round() == ta) * 100)

    print(f"\n{'='*65}")
    print(f"  Model: {model_name}   (n = {n} valid predictions)")
    print(f"{'='*65}")
    print(f"  {'Metric':<30} {'Valence':>12} {'Arousal':>12}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (↓ better)':<30} {mae_v:>12.3f} {mae_a:>12.3f}")
    print(f"  {'Pearson r (↑ better)':<30} {r_v:>12.3f} {r_a:>12.3f}")
    print(f"  {'p-value':<30} {p_v:>12.4f} {p_a:>12.4f}")
    print(f"  {'Accuracy ±1 (%)':<30} {acc_v:>11.1f}% {acc_a:>11.1f}%")
    print(f"  {'Exact match (%)':<30} {exact_v:>11.1f}% {exact_a:>11.1f}%")
    print(f"  {'Both within ±1 (%)':<30} {acc_both:>11.1f}%")

    return {
        'model':                    model_name,
        'n_valid':                  n,
        'mae_valence':              round(mae_v, 3),
        'mae_arousal':              round(mae_a, 3),
        'pearson_r_valence':        round(r_v, 3),
        'pearson_p_valence':        round(p_v, 4),
        'pearson_r_arousal':        round(r_a, 3),
        'pearson_p_arousal':        round(p_a, 4),
        'accuracy_v_within1_pct':  round(acc_v, 1),
        'accuracy_a_within1_pct':  round(acc_a, 1),
        'accuracy_both_within1_pct': round(acc_both, 1),
        'exact_match_v_pct':        round(exact_v, 1),
        'exact_match_a_pct':        round(exact_a, 1),
    }


def per_participant_breakdown(df, model_name):
    """Print per-participant MAE breakdown."""
    df = df.copy()
    df['inferred_valence']    = pd.to_numeric(df['inferred_valence'],    errors='coerce')
    df['inferred_arousal']    = pd.to_numeric(df['inferred_arousal'],    errors='coerce')
    df['self_reported_valence'] = pd.to_numeric(df['self_reported_valence'], errors='coerce')
    df['self_reported_arousal'] = pd.to_numeric(df['self_reported_arousal'], errors='coerce')

    print(f"\n  Per-participant MAE ({model_name}):")
    print(f"  {'Participant':<20} {'V_MAE':>8} {'A_MAE':>8} {'n':>6}")
    print(f"  {'-'*45}")

    for p, g in df.groupby('participant'):
        g = g.dropna(subset=['inferred_valence', 'inferred_arousal',
                             'self_reported_valence', 'self_reported_arousal'])
        if len(g) < 2:
            continue
        v_mae = np.mean(np.abs(g['inferred_valence'].astype(float) - g['self_reported_valence'].astype(float)))
        a_mae = np.mean(np.abs(g['inferred_arousal'].astype(float) - g['self_reported_arousal'].astype(float)))
        print(f"  {p:<20} {v_mae:>8.3f} {a_mae:>8.3f} {len(g):>6}")


def per_clip_analysis(dfs_with_names):
    """
    Average predictions per clip across participants for all models.
    Follows check_reported_vs_inferred.py approach.
    """
    results_list = []
    for df, mname in dfs_with_names:
        df = df.copy()
        df['clip_id']              = clean_id(df['clip_title'])
        df['inferred_valence']     = pd.to_numeric(df['inferred_valence'],     errors='coerce')
        df['inferred_arousal']     = pd.to_numeric(df['inferred_arousal'],     errors='coerce')
        df['self_reported_valence']= pd.to_numeric(df['self_reported_valence'],errors='coerce')
        df['self_reported_arousal']= pd.to_numeric(df['self_reported_arousal'],errors='coerce')

        clip_avg = df.groupby('clip_id').agg({
            'self_reported_valence': 'mean',
            'self_reported_arousal': 'mean',
            'inferred_valence':      'mean',
            'inferred_arousal':      'mean',
        }).rename(columns={
            'self_reported_valence': 'true_v',
            'self_reported_arousal': 'true_a',
            'inferred_valence':      f'{mname}_pred_v',
            'inferred_arousal':      f'{mname}_pred_a',
        })
        results_list.append(clip_avg)

    if results_list:
        # Start with first df (includes true_v, true_a), then join prediction columns from rest
        combined = results_list[0]
        for extra in results_list[1:]:
            pred_cols = [c for c in extra.columns if '_pred_' in c]
            combined = combined.join(extra[pred_cols], how='outer')
        combined.to_csv('per_clip_comparison.csv')
        print(f"\n  Per-clip averages saved to: per_clip_comparison.csv")
        print(f"  Total unique clips: {len(combined)}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

print("="*65)
print("  MODEL COMPARISON: Llama vs Mistral vs Llama4")
print("  Evaluation against self-reported induced emotion labels")
print("="*65)

model_configs = [
    ("Llama-3.3-70B",   "llm_results_llama.csv"),
    ("Mixtral-8x7B",    "llm_results_mistral.csv"),
    ("Llama-4",         "llm_results_llama4.csv"),
]

all_metrics = []
loaded_dfs  = {}

for model_name, results_file in model_configs:
    if not os.path.exists(results_file):
        print(f"\n⚠️  {results_file} not found. Skipping.")
        continue

    df = pd.read_csv(results_file)
    loaded_dfs[model_name] = df

    metrics = evaluate_model(df, model_name)
    if metrics:
        all_metrics.append(metrics)
        per_participant_breakdown(df, model_name)

# Per-clip analysis for all loaded models
if loaded_dfs:
    per_clip_analysis([(df, name) for name, df in loaded_dfs.items()])

# Save summary
if all_metrics:
    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv('model_comparison_results.csv', index=False)

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"\n  {'Model':<20} {'V_MAE':>8} {'A_MAE':>8} {'V_r':>8} {'A_r':>8}")
    print(f"  {'-'*50}")
    for _, r in summary_df.iterrows():
        print(f"  {r['model']:<20} {r['mae_valence']:>8.3f} {r['mae_arousal']:>8.3f} "
              f"{r['pearson_r_valence']:>8.3f} {r['pearson_r_arousal']:>8.3f}")

    best_v = summary_df.loc[summary_df['mae_valence'].idxmin(), 'model']
    best_a = summary_df.loc[summary_df['mae_arousal'].idxmin(), 'model']
    print(f"\n  Best model (Valence MAE):  {best_v}")
    print(f"  Best model (Arousal MAE):  {best_a}")
    print(f"\n  Files saved:")
    print(f"    model_comparison_results.csv")
    print(f"    per_clip_comparison.csv")

    # Guidance for next step
    min_mae = summary_df['mae_valence'].min()
    print(f"\n  ── Next Step Decision ────────────────────────────────")
    if min_mae > 1.0:
        print(f"  ⚠️  Best valence MAE = {min_mae:.3f} > 1.0")
        print(f"  → Consider K-EmoCon fine-tuning (10% subset)")
        print(f"  → Run kemocon_llm_inference_final.py to validate first")
    else:
        print(f"  ✅ Best valence MAE = {min_mae:.3f} ≤ 1.0")
        print(f"  → Zero-shot performance is acceptable")
        print(f"  → Proceed to K-EmoCon cross-dataset validation")

