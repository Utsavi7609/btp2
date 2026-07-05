"""Task A: I1-only switching analysis — stronger stats without I2 dependency."""
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
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

QUADS = ["Q1", "Q2", "Q3", "Q4"]

def load():
    df = pd.read_csv(os.path.join(BASE, "_consecutive_pair_dataset_AUDIO.csv"))
    valid = df[df["switching"].notna()].copy()
    valid["switching"] = valid["switching"].astype(bool)
    return df, valid

def part_a1(valid):
    """Per-quadrant drift heatmaps."""
    print("\n=== PART A1: Per-Quadrant Drift Heatmaps ===")
    for metric, label in [("delta_I1_val", "valence"), ("delta_I1_aro", "arousal")]:
        data = np.full((2, 4), np.nan)
        for j, q in enumerate(QUADS):
            for i, sw in enumerate([True, False]):
                sub = valid[(valid["clip1_E_quad"] == q) & (valid["switching"] == sw)]
                if len(sub) > 0:
                    data[i, j] = sub[metric].mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data))) if not np.all(np.isnan(data)) else 1
        sns.heatmap(pd.DataFrame(data, index=["Switching", "Non-Switching"], columns=QUADS),
                    annot=True, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, fmt=".3f", ax=ax)
        ax.set_title(f"Mean Delta I1 {label.capitalize()} by Starting Quadrant")
        ax.set_xlabel("Clip 1 E Quadrant")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, f"_per_quadrant_drift_heatmap_{label}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved _per_quadrant_drift_heatmap_{label}.png")

def part_a2(valid):
    """Per-transition drift (12 switching types)."""
    print("\n=== PART A2: Per-Transition Drift ===")
    rows = []
    for q1 in QUADS:
        for q2 in QUADS:
            if q1 == q2:
                continue
            sub = valid[(valid["clip1_E_quad"] == q1) & (valid["clip2_E_quad"] == q2) & (valid["switching"] == True)]
            if len(sub) > 0:
                rows.append({
                    "from": q1, "to": q2, "n": len(sub),
                    "mean_delta_I1_val": sub["delta_I1_val"].mean(),
                    "mean_delta_I1_aro": sub["delta_I1_aro"].mean(),
                    "frac_B1": (sub["clip2_B_I1_tau050"] == "B1").mean(),
                    "frac_B2": (sub["clip2_B_I1_tau050"] == "B2").mean(),
                })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(BASE, "_taskA_per_transition_drift.csv"), index=False)
    print(res.to_string(index=False))

    # 4x4 heatmap
    mat = np.full((4, 4), np.nan)
    for _, r in res.iterrows():
        i, j = QUADS.index(r["from"]), QUADS.index(r["to"])
        mat[i, j] = r["mean_delta_I1_val"]

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat))) if not np.all(np.isnan(mat)) else 2
    mask = np.eye(4, dtype=bool)
    sns.heatmap(pd.DataFrame(mat, index=QUADS, columns=QUADS), annot=True, cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax, fmt=".3f", ax=ax, mask=mask)
    ax.set_title("Mean Delta I1 Valence per Transition Type")
    ax.set_xlabel("Clip 2 E Quadrant (To)"); ax.set_ylabel("Clip 1 E Quadrant (From)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "_per_transition_drift_heatmap.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved _per_transition_drift_heatmap.png")

def part_a3(valid):
    """I1-only B1 Maintenance vs Drift per quadrant."""
    print("\n=== PART A3: I1-Only Maintenance vs Drift ===")
    rows = []
    for target_q in QUADS:
        b1_tq = valid[(valid["clip2_B_I1_tau050"] == "B1") & (valid["clip2_E_quad"] == target_q)]
        drift = b1_tq[b1_tq["clip1_E_quad"] != target_q]
        maint = b1_tq[b1_tq["clip1_E_quad"] == target_q]

        t_val, p_val = np.nan, np.nan
        u_val, p_u = np.nan, np.nan
        if len(drift) >= 2 and len(maint) >= 2:
            t_val, p_val = stats.ttest_ind(drift["delta_I1_val"], maint["delta_I1_val"])
            u_val, p_u = stats.mannwhitneyu(drift["delta_I1_val"], maint["delta_I1_val"],
                                            alternative="two-sided")

        rows.append({
            "target_quad": target_q,
            "n_drift": len(drift), "n_maint": len(maint),
            "mean_delta_drift": drift["delta_I1_val"].mean() if len(drift) > 0 else np.nan,
            "mean_delta_maint": maint["delta_I1_val"].mean() if len(maint) > 0 else np.nan,
            "ttest_t": t_val, "ttest_p": p_val,
            "mannwhitney_U": u_val, "mannwhitney_p": p_u,
        })
        print(f"  {target_q}: n_drift={len(drift)}, n_maint={len(maint)}, "
              f"t_p={p_val:.4f}, mw_p={p_u:.4f}" if not np.isnan(p_val) else
              f"  {target_q}: n_drift={len(drift)}, n_maint={len(maint)}, insufficient data")

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "_taskA_maintenance_drift_per_quadrant.csv"), index=False)

def part_a4(valid):
    """B2 dominance analysis."""
    print("\n=== PART A4: B2 Dominance Analysis ===")
    buckets = ["B1", "B2", "B3", "B4", "B5"]
    rows = []
    for bk in buckets:
        sw_sub = valid[(valid["clip1_B_I1_tau050"] == bk) & (valid["switching"] == True)]
        ns_sub = valid[(valid["clip1_B_I1_tau050"] == bk) & (valid["switching"] == False)]
        p_b2_sw = (sw_sub["clip2_B_I1_tau050"] == "B2").mean() if len(sw_sub) > 0 else np.nan
        p_b2_ns = (ns_sub["clip2_B_I1_tau050"] == "B2").mean() if len(ns_sub) > 0 else np.nan
        rows.append({"start_bucket": bk, "P_B2_switching": p_b2_sw, "n_sw": len(sw_sub),
                     "P_B2_nonswitching": p_b2_ns, "n_ns": len(ns_sub)})
        print(f"  {bk}: P(B2|sw)={p_b2_sw:.3f}(n={len(sw_sub)}), P(B2|ns)={p_b2_ns:.3f}(n={len(ns_sub)})")

    # Overall t-test
    a = (valid[valid["switching"] == True]["clip2_B_I1_tau050"] == "B2").astype(int)
    b = (valid[valid["switching"] == False]["clip2_B_I1_tau050"] == "B2").astype(int)
    t, p = stats.ttest_ind(a, b)
    print(f"\n  Overall B2 t-test: t={t:.4f}, p={p:.6f}, sig={p<0.05}")
    rows.append({"start_bucket": "ALL", "P_B2_switching": a.mean(), "n_sw": len(a),
                 "P_B2_nonswitching": b.mean(), "n_ns": len(b)})

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "_taskA_b2_dominance.csv"), index=False)

    # Bar chart
    res = pd.DataFrame(rows[:-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(buckets))
    w = 0.35
    ax.bar([i - w/2 for i in x], res["P_B2_switching"], w, label="Switching", color="#FF7043")
    ax.bar([i + w/2 for i in x], res["P_B2_nonswitching"], w, label="Non-Switching", color="#42A5F5")
    ax.set_xticks(list(x)); ax.set_xticklabels(buckets)
    ax.set_ylabel("P(Clip2 = B2)"); ax.set_title("B2 Dominance by Starting Bucket")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "_b2_dominance_analysis.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved _b2_dominance_analysis.png")

def part_a5(valid):
    """Hyp-1 retest with Mann-Whitney U."""
    print("\n=== PART A5: Hyp-1 Retest ===")
    rows = []
    for tau_str in ["025", "050", "075", "100"]:
        col = f"clip2_B_I1_tau{tau_str}"
        a = (valid[valid["switching"] == True][col] == "B1").astype(int)
        b = (valid[valid["switching"] == False][col] == "B1").astype(int)
        t, p_t = stats.ttest_ind(a, b)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"tau": tau_str, "ttest_t": t, "ttest_p": p_t,
                     "mannwhitney_U": u, "mannwhitney_p": p_u,
                     "sigma_sw": a.mean(), "sigma_ns": b.mean(),
                     "n_sw": len(a), "n_ns": len(b)})
        print(f"  tau={tau_str}: t_p={p_t:.4f}, mw_p={p_u:.4f}, sw={a.mean():.4f}, ns={b.mean():.4f}")

    # Per-quadrant at tau=050
    print("\n  Per-quadrant Hyp-1 at tau=050:")
    for q in QUADS:
        sub = valid[valid["clip2_E_quad"] == q]
        a = (sub[sub["switching"] == True]["clip2_B_I1_tau050"] == "B1").astype(int)
        b = (sub[sub["switching"] == False]["clip2_B_I1_tau050"] == "B1").astype(int)
        if len(a) >= 10 and len(b) >= 10:
            u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
            print(f"    {q}: mw_p={p_u:.4f}, sw={a.mean():.3f}(n={len(a)}), ns={b.mean():.3f}(n={len(b)})")
        else:
            print(f"    {q}: n_too_small (n_sw={len(a)}, n_ns={len(b)})")

    pd.DataFrame(rows).to_csv(os.path.join(BASE, "_taskA_hyp1_retest.csv"), index=False)

if __name__ == "__main__":
    df, valid = load()
    part_a1(valid)
    part_a2(valid)
    part_a3(valid)
    part_a4(valid)
    part_a5(valid)
    print("\n=== Task A complete. ===")
