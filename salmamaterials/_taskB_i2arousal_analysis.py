"""Task B: I2 arousal-only analysis — bypass valence compression."""
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = r"d:\BTP\btp2\salmamaterials"
PLOT_DIR = os.path.join(BASE, "_plots", "AUDIO")
QUADS = ["Q1", "Q2", "Q3", "Q4"]
sns.set_theme(style="whitegrid")

def load():
    df = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_AUDIO.csv"))
    valid = df[df["switching"].notna()].copy()
    valid["switching"] = valid["switching"].astype(bool)
    return df, valid

def part_b1(valid):
    """Arousal classification."""
    print("\n=== PART B1: I2 Arousal Classification ===")
    valid["I2_aro_class"] = np.where(valid["clip2_I2_aro"] > 3.0, "High", "Low")
    print(f"  High: {(valid['I2_aro_class']=='High').sum()}")
    print(f"  Low:  {(valid['I2_aro_class']=='Low').sum()}")
    return valid

def part_b2(valid):
    """I2 arousal by switching, per quadrant."""
    print("\n=== PART B2: I2 Arousal by Switching per Quadrant ===")
    rows = []
    for q in QUADS:
        sub = valid[valid["clip1_E_quad"] == q]
        sw = sub[sub["switching"] == True]["clip2_I2_aro"]
        ns = sub[sub["switching"] == False]["clip2_I2_aro"]
        t, p = (np.nan, np.nan)
        if len(sw) >= 10 and len(ns) >= 10:
            t, p = stats.ttest_ind(sw, ns)
        rows.append({
            "clip1_E_quad": q,
            "mean_I2aro_switching": sw.mean() if len(sw) > 0 else np.nan,
            "n_switching": len(sw),
            "mean_I2aro_nonswitching": ns.mean() if len(ns) > 0 else np.nan,
            "n_nonswitching": len(ns),
            "ttest_t": t, "ttest_p": p, "significant": p < 0.05 if not np.isnan(p) else False
        })
        print(f"  {q}: sw={sw.mean():.3f}(n={len(sw)}), ns={ns.mean():.3f}(n={len(ns)}), p={p:.4f}" if not np.isnan(p) else f"  {q}: insufficient data")

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "_taskB_arousal_by_switching.csv"), index=False)

    # Boxplot
    sub = valid[valid["clip1_E_quad"].isin(QUADS)]
    if len(sub) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=sub, x="clip1_E_quad", y="clip2_I2_aro", hue="switching",
                    palette="Set2", ax=ax, order=QUADS)
        ax.set_title("Clip2 I2 Arousal by Starting Quadrant and Switching")
        ax.set_ylabel("Clip2 I2 Arousal")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "_i2_arousal_by_switching.png"), bbox_inches="tight")
        plt.close(fig)
        print("  Saved _i2_arousal_by_switching.png")

def part_b3(valid):
    """Arousal carryover: correlation between clip1 E arousal and clip2 I2 arousal."""
    print("\n=== PART B3: Arousal Carryover (Correlation) ===")
    rows = []
    for sw_val, label in [(True, "Switching"), (False, "Non-Switching")]:
        sub = valid[valid["switching"] == sw_val].dropna(subset=["clip1_E_aro", "clip2_I2_aro"])
        if len(sub) >= 10:
            r, p = stats.pearsonr(sub["clip1_E_aro"], sub["clip2_I2_aro"])
            print(f"  {label}: Pearson r={r:.4f}, p={p:.6f} (n={len(sub)})")

            # Partial correlation controlling for clip2_E_aro
            sub2 = sub.dropna(subset=["clip2_E_aro"])
            if len(sub2) >= 10:
                # Partial correlation: regress out clip2_E_aro from both
                from numpy.linalg import lstsq
                x = sub2["clip2_E_aro"].values.reshape(-1, 1)
                x_aug = np.column_stack([x, np.ones(len(x))])
                res1 = sub2["clip1_E_aro"].values - x_aug @ lstsq(x_aug, sub2["clip1_E_aro"].values, rcond=None)[0]
                res2 = sub2["clip2_I2_aro"].values - x_aug @ lstsq(x_aug, sub2["clip2_I2_aro"].values, rcond=None)[0]
                pr, pp = stats.pearsonr(res1, res2)
                print(f"    Partial r (controlling clip2_E_aro): r={pr:.4f}, p={pp:.6f}")
            else:
                pr, pp = np.nan, np.nan

            rows.append({"group": label, "pearson_r": r, "pearson_p": p,
                        "partial_r": pr, "partial_p": pp, "n": len(sub)})

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "_taskB_arousal_carryover.csv"), index=False)

if __name__ == "__main__":
    df, valid = load()
    valid = part_b1(valid)
    part_b2(valid)
    part_b3(valid)
    print("\n=== Task B complete. ===")
