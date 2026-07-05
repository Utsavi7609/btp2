"""
_taskJ_physio_analysis.py
==========================
Physiological signal analysis using all 180 Empatica-covered clip rows.

Runs 4 tests that don't require pairing — uses every row with Empatica data:

  TEST A: Negative (Q2+Q3) vs Positive (Q1+Q4) clips  -> EDA and HR
          Hypothesis: negative clips produce higher sympathetic activation (EDA)
          and higher HR than positive clips.

  TEST B: High-Arousal (Q1+Q2) vs Low-Arousal (Q3+Q4) clips -> EDA and HR
          Hypothesis: high-arousal clips produce higher HR and EDA.

  TEST C: I2-E Quadrant Agreement vs Disagreement -> physio activation
          When I2 matches E (body confirmed the clip emotion), was EDA higher?
          Hypothesis: confirmed emotional responses show more physiological signal.

  TEST D: I2 vs I1 alignment with E (the Hyp-7 test)
          For each row: is I2 closer to E than I1 is?
          Fraction by E-quadrant — shows where body beats survey.

Run:  py -3 _taskJ_physio_analysis.py
Deps: pip install pandas numpy scipy
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

I2_CSV       = r"D:\BTP\btp2\salmamaterials\_validation_with_I2.csv"
EXPRESSED_CSV= r"D:\BTP\btp2\salmamaterials\final_expressed_valence_AUDIO.csv"
OUT_CSV      = r"D:\BTP\btp2\salmamaterials\_taskJ_physio_results.csv"

SEP  = "=" * 65
DASH = "-" * 65


def get_quad(v, a):
    try:
        v, a = float(v), float(a)
        if math.isnan(v) or math.isnan(a): return "Q0"
    except (TypeError, ValueError):
        return "Q0"
    if v > 3 and a > 3: return "Q1"
    if v < 3 and a > 3: return "Q2"
    if v < 3 and a < 3: return "Q3"
    if v > 3 and a < 3: return "Q4"
    return "Q0"


def dist(v1, a1, v2, a2):
    try:
        return math.sqrt((float(v1)-float(v2))**2 + (float(a1)-float(a2))**2)
    except Exception:
        return float("nan")


def mwu(a, b, label_a, label_b, metric):
    """Mann-Whitney U, print result, return p-value."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < 3 or len(b) < 3:
        print(f"    Insufficient data: {label_a} n={len(a)}, {label_b} n={len(b)}")
        return float("nan")
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    sig = "*** p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "n.s.")
    print(f"    {label_a:20s}: n={len(a):3d}  mean={a.mean():.3f}  median={a.median():.3f}")
    print(f"    {label_b:20s}: n={len(b):3d}  mean={b.mean():.3f}  median={b.median():.3f}")
    print(f"    Mann-Whitney ({metric}): U={stat:.0f}  p={p:.4f}  [{sig}]")
    return p


def main():
    print(SEP)
    print("  TASK-J: Direct Physiological Analysis (180 Empatica rows)")
    print(SEP)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(I2_CSV)
    e_df = pd.read_csv(EXPRESSED_CSV)
    e_dict = {str(r["clip_id"]): (float(r["expressed_valence"]),
                                   float(r["expressed_arousal"]))
              for _, r in e_df.iterrows()}

    # Keep only rows with Empatica physio data
    phys = df[df["phys_hr_mean"].notna() | df["phys_eda_mean"].notna()].copy()
    print(f"\n  Rows with Empatica data: {len(phys)}")
    print(f"  phys_hr_mean  non-null : {phys['phys_hr_mean'].notna().sum()}")
    print(f"  phys_eda_mean non-null : {phys['phys_eda_mean'].notna().sum()}")
    print(f"  I2_val        non-null : {phys['I2_val'].notna().sum()}")

    # Attach expressed emotion quadrant to each row
    def clip_equad(title):
        cid = str(title).replace(".mp4","").strip()
        e = e_dict.get(cid)
        if e is None: return "Q0"
        return get_quad(*e)

    phys = phys.copy()
    phys["E_quad"]  = phys["clip_title"].apply(clip_equad)
    phys["P_quad"]  = phys.apply(lambda r: get_quad(r["p_val"], r["p_aro"]), axis=1)
    phys["I1_quad"] = phys.apply(lambda r: get_quad(r["i1_val"], r["i1_aro"]), axis=1)
    phys["I2_quad"] = phys.apply(lambda r: get_quad(r["I2_val"], r["I2_aro"]), axis=1)

    # Valence polarity: negative = Q2+Q3 (v<3), positive = Q1+Q4 (v>3)
    phys["E_valence_neg"] = phys["E_quad"].isin(["Q2","Q3"])
    phys["E_valence_pos"] = phys["E_quad"].isin(["Q1","Q4"])

    # Arousal polarity: high = Q1+Q2 (a>3), low = Q3+Q4 (a<3)
    phys["E_arousal_hi"]  = phys["E_quad"].isin(["Q1","Q2"])
    phys["E_arousal_lo"]  = phys["E_quad"].isin(["Q3","Q4"])

    print(f"\n  E-quadrant distribution:")
    for q in ["Q1","Q2","Q3","Q4","Q0"]:
        n = (phys["E_quad"] == q).sum()
        print(f"    {q}: {n}")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # TEST A — Negative vs Positive clips: EDA and HR
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST A — Negative (Q2+Q3) vs Positive (Q1+Q4) clips")
    print(f"  Hypothesis: negative clips produce higher EDA (sympathetic activation)")
    print(f"              and higher HR than positive clips\n")

    neg = phys[phys["E_valence_neg"]]
    pos = phys[phys["E_valence_pos"]]

    print(f"  [A1] EDA:")
    p_a1 = mwu(neg["phys_eda_mean"], pos["phys_eda_mean"],
                "Negative (Q2+Q3)", "Positive (Q1+Q4)", "EDA")
    results["A1_eda_neg_vs_pos_p"] = round(p_a1, 4)

    print(f"\n  [A2] HR:")
    p_a2 = mwu(neg["phys_hr_mean"], pos["phys_hr_mean"],
                "Negative (Q2+Q3)", "Positive (Q1+Q4)", "HR")
    results["A2_hr_neg_vs_pos_p"] = round(p_a2, 4)

    # ══════════════════════════════════════════════════════════════════════
    # TEST B — High-Arousal vs Low-Arousal clips: EDA and HR
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST B — High-Arousal (Q1+Q2) vs Low-Arousal (Q3+Q4) clips")
    print(f"  Hypothesis: high-arousal clips produce higher HR and EDA\n")

    hi = phys[phys["E_arousal_hi"]]
    lo = phys[phys["E_arousal_lo"]]

    print(f"  [B1] EDA:")
    p_b1 = mwu(hi["phys_eda_mean"], lo["phys_eda_mean"],
                "High-Arousal (Q1+Q2)", "Low-Arousal  (Q3+Q4)", "EDA")
    results["B1_eda_hi_vs_lo_p"] = round(p_b1, 4)

    print(f"\n  [B2] HR:")
    p_b2 = mwu(hi["phys_hr_mean"], lo["phys_hr_mean"],
                "High-Arousal (Q1+Q2)", "Low-Arousal  (Q3+Q4)", "HR")
    results["B2_hr_hi_vs_lo_p"] = round(p_b2, 4)

    # ══════════════════════════════════════════════════════════════════════
    # TEST C — I2-E agreement vs disagreement: physio activation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST C — I2 agrees with E vs I2 disagrees with E")
    print(f"  Hypothesis: when body confirms clip emotion (I2==E_quad), EDA is higher\n")

    sub_c = phys[phys["I2_quad"].isin(["Q1","Q2","Q3","Q4"]) &
                 phys["E_quad"].isin(["Q1","Q2","Q3","Q4"])].copy()
    sub_c["I2_agrees_E"] = sub_c["I2_quad"] == sub_c["E_quad"]

    agree    = sub_c[sub_c["I2_agrees_E"]]
    disagree = sub_c[~sub_c["I2_agrees_E"]]
    print(f"    I2==E_quad : {len(agree)}   I2!=E_quad : {len(disagree)}")

    print(f"\n  [C1] EDA:")
    p_c1 = mwu(agree["phys_eda_mean"], disagree["phys_eda_mean"],
                "I2 agrees E", "I2 disagrees E", "EDA")
    results["C1_eda_agree_vs_disagree_p"] = round(p_c1, 4)

    print(f"\n  [C2] HR:")
    p_c2 = mwu(agree["phys_hr_mean"], disagree["phys_hr_mean"],
                "I2 agrees E", "I2 disagrees E", "HR")
    results["C2_hr_agree_vs_disagree_p"] = round(p_c2, 4)

    # ══════════════════════════════════════════════════════════════════════
    # TEST D — I2 vs I1: which is closer to E? (Hyp-7 body-beats-survey)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST D — Does body (I2) track expressed emotion better than survey (I1)?")
    print(f"  Method: Euclidean distance d(I2,E) < d(I1,E) for each row\n")

    sub_d = phys.dropna(subset=["I2_val","I2_aro","i1_val","i1_aro"]).copy()

    # Get expressed val/aro for each row
    def get_e(title):
        cid = str(title).replace(".mp4","").strip()
        return e_dict.get(cid, (float("nan"), float("nan")))

    sub_d["ev"]  = sub_d["clip_title"].apply(lambda t: get_e(t)[0])
    sub_d["ea"]  = sub_d["clip_title"].apply(lambda t: get_e(t)[1])
    sub_d = sub_d.dropna(subset=["ev","ea"])

    sub_d["d_I2_E"] = sub_d.apply(
        lambda r: dist(r["I2_val"], r["I2_aro"], r["ev"], r["ea"]), axis=1)
    sub_d["d_I1_E"] = sub_d.apply(
        lambda r: dist(r["i1_val"], r["i1_aro"], r["ev"], r["ea"]), axis=1)
    sub_d["I2_closer"] = sub_d["d_I2_E"] < sub_d["d_I1_E"]

    n_total    = len(sub_d)
    n_i2_wins  = int(sub_d["I2_closer"].sum())
    pct        = 100 * n_i2_wins / n_total if n_total else 0

    print(f"    Total rows with I2+I1+E: {n_total}")
    print(f"    I2 closer to E than I1 : {n_i2_wins}/{n_total}  ({pct:.1f}%)")

    # Wilcoxon signed-rank: d_I1_E > d_I2_E (I2 is systematically closer)
    diffs = sub_d["d_I1_E"] - sub_d["d_I2_E"]
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 5:
        stat_w, p_w = wilcoxon(diffs_nonzero, alternative="greater")
        sig = "*** p<0.05" if p_w < 0.05 else ("~ p<0.10" if p_w < 0.10 else "n.s.")
        print(f"    Wilcoxon signed-rank (d_I1>d_I2, i.e. I2 closer): "
              f"W={stat_w:.0f}  p={p_w:.4f}  [{sig}]")
        results["D_wilcoxon_I2_closer_p"] = round(p_w, 4)
    else:
        print(f"    Insufficient non-zero differences for Wilcoxon.")
        results["D_wilcoxon_I2_closer_p"] = float("nan")

    results["D_i2_wins_pct"] = round(pct, 2)

    # By E-quadrant breakdown
    print(f"\n    I2-closer rate by E-quadrant:")
    for q in ["Q1","Q2","Q3","Q4"]:
        sq = sub_d[sub_d["E_quad"] == q]
        if len(sq) == 0: continue
        wins = sq["I2_closer"].sum()
        print(f"      {q}: {wins}/{len(sq)}  ({100*wins/len(sq):.1f}%)  "
              f"mean d(I2,E)={sq['d_I2_E'].mean():.2f}  "
              f"mean d(I1,E)={sq['d_I1_E'].mean():.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(f"  {DASH}")
    tests = [
        ("A1", "EDA: Negative vs Positive clips",     results.get("A1_eda_neg_vs_pos_p")),
        ("A2", "HR:  Negative vs Positive clips",     results.get("A2_hr_neg_vs_pos_p")),
        ("B1", "EDA: High-Arousal vs Low-Arousal",    results.get("B1_eda_hi_vs_lo_p")),
        ("B2", "HR:  High-Arousal vs Low-Arousal",    results.get("B2_hr_hi_vs_lo_p")),
        ("C1", "EDA: I2==E vs I2!=E (confirmed)",     results.get("C1_eda_agree_vs_disagree_p")),
        ("C2", "HR:  I2==E vs I2!=E (confirmed)",     results.get("C2_hr_agree_vs_disagree_p")),
        ("D",  "I2 closer to E than I1 (Wilcoxon)",  results.get("D_wilcoxon_I2_closer_p")),
    ]
    for tid, desc, p in tests:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            status = "    N/A  "
        elif p < 0.05:
            status = "*** PASS"
        elif p < 0.10:
            status = "~   MARGINAL"
        else:
            status = "    n.s. "
        print(f"  [{tid}] {status}  p={str(round(p,4)) if p==p else 'nan':7s}  {desc}")

    print(SEP)

    # Save enriched physio rows
    phys.to_csv(OUT_CSV, index=False)
    print(f"\n  Enriched data -> {OUT_CSV}")


if __name__ == "__main__":
    main()
