"""Task E: Final verification checks."""
import pandas as pd
import numpy as np
import os

BASE = r"d:\BTP\btp2\salmamaterials"

def main():
    df = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_AUDIO.csv"))
    vid = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_VIDEO.csv"))

    print("=" * 70)
    print("CHECK 1: I2 Valence Compression")
    print("=" * 70)
    i2v = df["clip2_I2_val"].dropna()
    i2a = df["clip2_I2_aro"].dropna()
    print(f"  I2 Valence: min={i2v.min():.3f}, max={i2v.max():.3f}, mean={i2v.mean():.3f}, std={i2v.std():.3f}")
    print(f"  I2 Arousal: min={i2a.min():.3f}, max={i2a.max():.3f}, mean={i2a.mean():.3f}, std={i2a.std():.3f}")
    print(f"  Fraction I2_val < 3.0: {(i2v < 3.0).sum()}/{len(i2v)} = {(i2v < 3.0).mean():.3f}")
    print(f"  Fraction I2_val > 3.0: {(i2v > 3.0).sum()}/{len(i2v)} = {(i2v > 3.0).mean():.3f}")
    print(f"  Fraction I2_val == 3.0: {(i2v == 3.0).sum()}/{len(i2v)} = {(i2v == 3.0).mean():.3f}")

    print()
    print("=" * 70)
    print("CHECK 2: B2 Dominance")
    print("=" * 70)
    valid = df[df["switching"].notna()]
    sw_t = valid[valid["switching"] == True]
    sw_f = valid[valid["switching"] == False]
    b2_sw = (sw_t["clip2_B_I1_tau050"] == "B2").mean()
    b2_ns = (sw_f["clip2_B_I1_tau050"] == "B2").mean()
    print(f"  P(clip2=B2 | switching=True):  {b2_sw:.4f} (n={len(sw_t)})")
    print(f"  P(clip2=B2 | switching=False): {b2_ns:.4f} (n={len(sw_f)})")

    print()
    print("=" * 70)
    print("CHECK 3: VIDEO Issue")
    print("=" * 70)
    print(f"  VIDEO clip1_E_val: min={vid['clip1_E_val'].min():.3f}, max={vid['clip1_E_val'].max():.3f}")
    print(f"  VIDEO clip1_E_aro: min={vid['clip1_E_aro'].min():.3f}, max={vid['clip1_E_aro'].max():.3f}")
    q1_frac = (vid["clip1_E_quad"] == "Q1").mean()
    print(f"  Fraction Q1: {q1_frac:.4f} ({(vid['clip1_E_quad']=='Q1').sum()}/{len(vid)})")

    print()
    print("=" * 70)
    print("CHECK 4: Reproducibility of Significant Results")
    print("=" * 70)
    # Hyp-2: opposing switching, B3+B6
    OPPOSING = {("Q1","Q3"),("Q3","Q1"),("Q2","Q4"),("Q4","Q2")}
    opp_sw = sw_t[sw_t.apply(lambda r: (r["clip1_E_quad"], r["clip2_E_quad"]) in OPPOSING, axis=1)]
    hyp2_sigma = (opp_sw["clip2_combined_tau050"] == "B3+B6").mean() if len(opp_sw) > 0 else float("nan")
    print(f"  Hyp-2 recomputed sigma (opposing switching): {hyp2_sigma:.4f} (n={len(opp_sw)})")
    stored = pd.read_csv(os.path.join(BASE, "_sigma_results_AUDIO.csv"))
    h2_stored = stored[(stored["hyp_id"] == "Hyp-2_opposing") & (stored["tau"] == 50)]
    if len(h2_stored) > 0:
        print(f"  Hyp-2 stored sigma:                          {h2_stored.iloc[0]['sigma_switching']:.4f}")
        diff = abs(hyp2_sigma - h2_stored.iloc[0]["sigma_switching"])
        print(f"  Difference: {diff:.6f} {'PASS' if diff < 0.001 else 'FAIL'}")

    # Hyp-9: mirror rate
    neut = df[(df["clip2_E_val"] - 3.0).abs().le(0.5) & (df["clip2_E_aro"] - 3.0).abs().le(0.5)]
    b2_neut = neut[neut["clip2_B_I1_tau050"] == "B2"]
    if len(b2_neut) > 0:
        d1 = np.sqrt((b2_neut["clip2_P_val"] - b2_neut["clip1_I1_val"])**2 +
                     (b2_neut["clip2_P_aro"] - b2_neut["clip1_I1_aro"])**2)
        d2 = np.sqrt((b2_neut["clip2_P_val"] - b2_neut["clip2_E_val"])**2 +
                     (b2_neut["clip2_P_aro"] - b2_neut["clip2_E_aro"])**2)
        mirror = (d1 < d2)
        sw_mir = mirror.loc[b2_neut[b2_neut["switching"] == True].index]
        ns_mir = mirror.loc[b2_neut[b2_neut["switching"] == False].index]
        print(f"\n  Hyp-9 recomputed mirror_rate (switching): {sw_mir.mean():.4f} (n={len(sw_mir)})")
        print(f"  Hyp-9 recomputed mirror_rate (nonswitching): {ns_mir.mean():.4f} (n={len(ns_mir)})")
        h9_stored = stored[(stored["hyp_id"] == "Hyp-9_relaxed") & (stored["tau"] == 50)]
        if len(h9_stored) > 0:
            diff_sw = abs(sw_mir.mean() - h9_stored.iloc[0]["sigma_switching"])
            print(f"  Stored sw: {h9_stored.iloc[0]['sigma_switching']:.4f}, diff: {diff_sw:.6f} {'PASS' if diff_sw < 0.001 else 'FAIL'}")

    print()
    print("=" * 70)
    print("CHECK 5: Missing Data Audit (AUDIO)")
    print("=" * 70)
    for col in ["clip2_I1_val", "clip2_I2_val", "clip2_P_val", "clip2_E_val",
                "clip2_I1_aro", "clip2_I2_aro", "clip2_P_aro", "clip2_E_aro"]:
        nan_count = df[col].isna().sum()
        print(f"  {col}: {nan_count}/{len(df)} NaN ({nan_count/len(df)*100:.1f}%)")

if __name__ == "__main__":
    main()
