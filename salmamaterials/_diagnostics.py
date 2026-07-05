"""Diagnostic script for all 5 assessment action items."""
import pandas as pd
import numpy as np

BASE = r"d:\BTP\btp2\salmamaterials"

print("=" * 70)
print("ACTION 1: Key Hypothesis Rows from _statistical_tests_AUDIO.csv")
print("=" * 70)
st = pd.read_csv(f"{BASE}\\_statistical_tests_AUDIO.csv")
for hyp in ["Hyp-1", "Hyp-7", "Hyp-8_full", "Hyp-8_aro", "Hyp-4_chaos_vs_baseline", "Maintenance_vs_Drift"]:
    row = st[(st["hyp_id"] == hyp) & (st["tau"] == 50)]
    if len(row) > 0:
        r = row.iloc[0]
        t = r["t_stat"]
        p = r["p_value"]
        sig = r["significant"]
        sw = r["sigma_switching"]
        nsw = r["n_switching"]
        ns = r["sigma_nonswitching"]
        nns = r["n_nonswitching"]
        dm = r["effect_direction_matches"]
        notes = r["notes"]
        print(f"  {hyp}: t={t:.4f}, p={p:.6f}, sig={sig}, sw={sw:.4f}(n={int(nsw)}), ns={ns:.4f}(n={int(nns)}), dir_match={dm}, notes={notes}")
    else:
        print(f"  {hyp}: NOT FOUND at tau=50")

print()
print("=" * 70)
print("ACTION 2: AUDIO switching=True combined bucket top 15 (tau=050)")
print("=" * 70)
pair = pd.read_csv(f"{BASE}\\_consecutive_pair_dataset_AUDIO.csv")
sw = pair[pair["switching"] == True]
print(sw["clip2_combined_tau050"].value_counts().head(15).to_string())
b3b6 = (sw["clip2_combined_tau050"] == "B3+B6").sum()
print(f"\nTotal switching rows: {len(sw)}")
print(f"B3+B6 count in switching: {b3b6}")

print()
print("=" * 70)
print("ACTION 3: VIDEO E distribution")
print("=" * 70)
vid = pd.read_csv(f"{BASE}\\_consecutive_pair_dataset_VIDEO.csv")
print("Clip1 E Quadrant:")
print(vid["clip1_E_quad"].value_counts().to_string())
print(f"clip1_E_val: min={vid['clip1_E_val'].min():.3f}, max={vid['clip1_E_val'].max():.3f}, mean={vid['clip1_E_val'].mean():.3f}")
print(f"clip1_E_aro: min={vid['clip1_E_aro'].min():.3f}, max={vid['clip1_E_aro'].max():.3f}, mean={vid['clip1_E_aro'].mean():.3f}")
print(f"clip1_E_val <= 3.0: {(vid['clip1_E_val'] <= 3.0).sum()}")
print(f"clip1_E_aro <= 3.0: {(vid['clip1_E_aro'] <= 3.0).sum()}")

print()
print("=" * 70)
print("ACTION 4: Hyp-9 and Hyp-4 n values from _sigma_results_AUDIO.csv")
print("=" * 70)
sig = pd.read_csv(f"{BASE}\\_sigma_results_AUDIO.csv")
for prefix in ["Hyp-9", "Hyp-4"]:
    rows = sig[(sig["hyp_id"].str.startswith(prefix)) & (sig["tau"] == 50)]
    print(f"\n  {prefix} at tau=50:")
    for _, r in rows.iterrows():
        print(f"    {r['hyp_id']}: sw={r['sigma_switching']:.3f}(n={int(r['n_switching'])}), ns={r['sigma_nonswitching']:.3f}(n={int(r['n_nonswitching'])}), all={r['sigma_all']:.3f}(n={int(r['n_all'])})")

print()
print("=" * 70)
print("EXTRA: Hyp-1 and Hyp-7 across ALL tau for AUDIO")
print("=" * 70)
for hyp in ["Hyp-1", "Hyp-7"]:
    rows = sig[sig["hyp_id"] == hyp]
    print(f"\n  {hyp}:")
    for _, r in rows.iterrows():
        print(f"    tau={int(r['tau']):03d}: sw={r['sigma_switching']:.4f}(n={int(r['n_switching'])}), ns={r['sigma_nonswitching']:.4f}(n={int(r['n_nonswitching'])})")
