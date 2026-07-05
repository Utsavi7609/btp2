"""
_taskK_multisensor_final.py
============================
Final multi-sensor physiological analysis using ALL available Empatica signals:
  - Pulse Rate (HR)          from D:\BTP\btp2main\2lf92x651f5z6t6y
  - EDA                      from D:\BTP\btp2main\2lf92x651f5z6t6y
  - Skin Temperature         from D:\BTP\btp2main\3q4dw226h3b2y6ye
  - Respiratory Rate         from D:\BTP\btp2main\3q4dw226h3b2y6ye  (sparse, best-effort)

Tests run on 180 Empatica-covered clip rows:
  TEST A : Negative (Q2+Q3) vs Positive (Q1+Q4) clips  [EDA, HR, Temp]
  TEST B : High-Arousal (Q1+Q2) vs Low-Arousal (Q3+Q4) [EDA, HR, Temp]
  TEST C : I2 agrees E vs disagrees E                   [EDA, HR, Temp]
  TEST D : I2 closer to E than I1? (Hyp-7 proxy)        [Wilcoxon]
  TEST E : Switching (Drift) vs Non-Switching (Maint)   [EDA, HR, Temp, |I1-I2|]
           (Hyp-8: I1-I2 valence divergence)

Run:  py -3 _taskK_multisensor_final.py
"""

import math, glob, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

# ── PATHS ──────────────────────────────────────────────────────────────────
I2_CSV        = r"D:\BTP\btp2\salmamaterials\_validation_with_I2.csv"
EXPRESSED_CSV = r"D:\BTP\btp2\salmamaterials\final_expressed_valence_AUDIO.csv"
OUT_CSV       = r"D:\BTP\btp2\salmamaterials\_taskK_multisensor_results.csv"

EMPATICA_ORIG = r"D:\BTP\btp2main\2lf92x651f5z6t6y"   # HR, EDA, PRV
EMPATICA_NEW  = r"D:\BTP\btp2main\3q4dw226h3b2y6ye"   # Temperature, Resp-rate

WINDOW_PAD_MS = 60_000   # 60-s left pad (same as _taskH)

SEP  = "=" * 70
DASH = "-" * 70


# ── SIGNAL LOADER ──────────────────────────────────────────────────────────

def load_signal(base_dir: str, suffix: str) -> pd.DataFrame:
    pattern = os.path.join(base_dir, "**", f"*_{suffix}.csv")
    files   = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return pd.DataFrame(columns=["timestamp_unix", "value"])
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["missing_value_reason"].isna()].copy()
        meta = {"timestamp_unix", "timestamp_iso", "participant_full_id", "missing_value_reason"}
        val_cols = [c for c in df.columns if c not in meta]
        if not val_cols:
            continue
        df = df[["timestamp_unix", val_cols[0]]].rename(columns={val_cols[0]: "value"})
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["timestamp_unix", "value"])
    out = pd.concat(dfs, ignore_index=True)
    out["timestamp_unix"] = out["timestamp_unix"].astype(np.int64)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).sort_values("timestamp_unix").reset_index(drop=True)
    t0 = pd.to_datetime(out["timestamp_unix"].iloc[0],  unit="ms", utc=True).strftime("%m-%d %H:%M")
    t1 = pd.to_datetime(out["timestamp_unix"].iloc[-1], unit="ms", utc=True).strftime("%m-%d %H:%M")
    print(f"  {suffix:20s}: {len(out):4d} valid rows  ({t0} -> {t1} UTC)")
    return out


def window_mean(sig_df: pd.DataFrame, t_start_ms: int, t_end_ms: int) -> float:
    sub = sig_df[
        (sig_df["timestamp_unix"] >= t_start_ms - WINDOW_PAD_MS) &
        (sig_df["timestamp_unix"] <= t_end_ms)
    ]["value"]
    return float(sub.mean()) if len(sub) > 0 else float("nan")


# ── HELPERS ────────────────────────────────────────────────────────────────

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


def euc(v1, a1, v2, a2):
    try:
        return math.sqrt((float(v1)-float(v2))**2 + (float(a1)-float(a2))**2)
    except Exception:
        return float("nan")


def mwu(a, b, label_a, label_b, metric, alt="two-sided"):
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        print(f"    Insufficient data: {label_a} n={len(a)}, {label_b} n={len(b)}")
        return float("nan")
    stat, p = mannwhitneyu(a, b, alternative=alt)
    sig = "*** p<0.05" if p < 0.05 else ("~   p<0.10" if p < 0.10 else "    n.s. ")
    print(f"    {label_a:25s}: n={len(a):3d}  mean={a.mean():.3f}  median={a.median():.3f}")
    print(f"    {label_b:25s}: n={len(b):3d}  mean={b.mean():.3f}  median={b.median():.3f}")
    print(f"    MWU ({metric}): U={stat:.0f}  p={p:.4f}  [{sig}]")
    return p


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  TASK-K: Multi-Sensor Final Physiological Analysis")
    print(SEP)

    # 1. Load all signals
    print("\n  Loading Empatica signals ...")
    hr_df   = load_signal(EMPATICA_ORIG, "pulse-rate")
    eda_df  = load_signal(EMPATICA_ORIG, "eda")
    temp_df = load_signal(EMPATICA_NEW,  "temperature")
    resp_df = load_signal(EMPATICA_NEW,  "respiratory-rate")

    phys_start = hr_df["timestamp_unix"].min() if len(hr_df) else 0
    phys_end   = hr_df["timestamp_unix"].max() if len(hr_df) else 0

    # 2. Load validation CSV and expressed emotion
    df = pd.read_csv(I2_CSV)
    e_df = pd.read_csv(EXPRESSED_CSV)
    e_dict = {str(r["clip_id"]): (float(r["expressed_valence"]), float(r["expressed_arousal"]))
              for _, r in e_df.iterrows()}

    # 3. Keep only Empatica-covered rows
    phys = df[
        (df["timestamp_clip_end"]   >= phys_start - WINDOW_PAD_MS) &
        (df["timestamp_clip_start"] <= phys_end)
    ].copy()
    print(f"\n  Rows in Empatica window: {len(phys)}")

    # 4. Extract fresh window stats for all 3 sensors per row
    print("  Extracting per-clip window stats ...")
    hrs, edas, temps, resps = [], [], [], []
    for _, row in phys.iterrows():
        ts, te = int(row["timestamp_clip_start"]), int(row["timestamp_clip_end"])
        hrs.append(window_mean(hr_df,   ts, te))
        edas.append(window_mean(eda_df,  ts, te))
        temps.append(window_mean(temp_df, ts, te))
        resps.append(window_mean(resp_df, ts, te))

    phys = phys.copy()
    phys["hr"]   = hrs
    phys["eda"]  = edas
    phys["temp"] = temps
    phys["resp"] = resps

    # 5. Quadrant labels
    def clip_equad(title):
        cid = str(title).replace(".mp4","").strip()
        e = e_dict.get(cid)
        return get_quad(*e) if e else "Q0"

    phys["E_quad"]  = phys["clip_title"].apply(clip_equad)
    phys["I2_quad"] = phys.apply(lambda r: get_quad(r["I2_val"], r["I2_aro"]), axis=1)
    phys["I1_quad"] = phys.apply(lambda r: get_quad(r["i1_val"], r["i1_aro"]), axis=1)

    # expressed val/aro columns for distance tests
    phys["ev"] = phys["clip_title"].apply(
        lambda t: e_dict.get(str(t).replace(".mp4","").strip(), (float("nan"),float("nan")))[0])
    phys["ea"] = phys["clip_title"].apply(
        lambda t: e_dict.get(str(t).replace(".mp4","").strip(), (float("nan"),float("nan")))[1])

    print(f"\n  Sensor coverage (non-null):")
    print(f"    HR    : {phys['hr'].notna().sum():3d} / {len(phys)}")
    print(f"    EDA   : {phys['eda'].notna().sum():3d} / {len(phys)}")
    print(f"    Temp  : {phys['temp'].notna().sum():3d} / {len(phys)}")
    print(f"    Resp  : {phys['resp'].notna().sum():3d} / {len(phys)}")

    print(f"\n  E-quad distribution:")
    for q in ["Q1","Q2","Q3","Q4","Q0"]:
        print(f"    {q}: {(phys['E_quad']==q).sum()}")

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # TEST A — Negative (Q2+Q3) vs Positive (Q1+Q4)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST A — Clip Valence: Negative (Q2+Q3) vs Positive (Q1+Q4)")
    print("  Hypothesis: negative -> lower skin temp, higher EDA, higher HR\n")

    neg = phys[phys["E_quad"].isin(["Q2","Q3"])]
    pos = phys[phys["E_quad"].isin(["Q1","Q4"])]

    print("  [A1] Skin Temperature  (expect: neg < pos — vasoconstriction):")
    p = mwu(neg["temp"], pos["temp"], "Negative (Q2+Q3)", "Positive (Q1+Q4)", "Temp°C")
    results["A1_temp_neg_vs_pos_p"] = round(p, 4)

    print("\n  [A2] EDA:")
    p = mwu(neg["eda"], pos["eda"], "Negative (Q2+Q3)", "Positive (Q1+Q4)", "EDA")
    results["A2_eda_neg_vs_pos_p"] = round(p, 4)

    print("\n  [A3] HR:")
    p = mwu(neg["hr"], pos["hr"], "Negative (Q2+Q3)", "Positive (Q1+Q4)", "HR")
    results["A3_hr_neg_vs_pos_p"] = round(p, 4)

    # ══════════════════════════════════════════════════════════════════════
    # TEST B — High-Arousal (Q1+Q2) vs Low-Arousal (Q3+Q4)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST B — Clip Arousal: High (Q1+Q2) vs Low (Q3+Q4)")
    print("  Hypothesis: high-arousal -> higher HR, higher resp rate\n")

    hi = phys[phys["E_quad"].isin(["Q1","Q2"])]
    lo = phys[phys["E_quad"].isin(["Q3","Q4"])]

    print("  [B1] Skin Temperature:")
    p = mwu(hi["temp"], lo["temp"], "High-Arousal (Q1+Q2)", "Low-Arousal (Q3+Q4)", "Temp°C")
    results["B1_temp_hi_vs_lo_p"] = round(p, 4)

    print("\n  [B2] HR:")
    p = mwu(hi["hr"], lo["hr"], "High-Arousal (Q1+Q2)", "Low-Arousal (Q3+Q4)", "HR")
    results["B2_hr_hi_vs_lo_p"] = round(p, 4)

    print("\n  [B3] EDA:")
    p = mwu(hi["eda"], lo["eda"], "High-Arousal (Q1+Q2)", "Low-Arousal (Q3+Q4)", "EDA")
    results["B3_eda_hi_vs_lo_p"] = round(p, 4)

    # Respiratory rate (sparse, best-effort)
    hi_resp = hi["resp"].dropna()
    lo_resp = lo["resp"].dropna()
    if len(hi_resp) >= 3 and len(lo_resp) >= 3:
        print("\n  [B4] Respiratory Rate:")
        p = mwu(hi["resp"], lo["resp"], "High-Arousal (Q1+Q2)", "Low-Arousal (Q3+Q4)", "RespRate")
        results["B4_resp_hi_vs_lo_p"] = round(p, 4)
    else:
        print(f"\n  [B4] Resp-rate: too sparse (hi n={len(hi_resp)}, lo n={len(lo_resp)}) — skip")
        results["B4_resp_hi_vs_lo_p"] = float("nan")

    # ══════════════════════════════════════════════════════════════════════
    # TEST C — I2 agrees with E vs disagrees
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST C — I2 agrees with E (I2_quad == E_quad) vs disagrees")
    print("  Hypothesis: confirmed physiological inference -> lower HR (unambiguous signal)\n")

    sub_c = phys[phys["I2_quad"].isin(["Q1","Q2","Q3","Q4"]) &
                 phys["E_quad"].isin(["Q1","Q2","Q3","Q4"])].copy()
    sub_c["agrees"] = sub_c["I2_quad"] == sub_c["E_quad"]
    agree    = sub_c[sub_c["agrees"]]
    disagree = sub_c[~sub_c["agrees"]]
    print(f"    I2==E: {len(agree)}   I2!=E: {len(disagree)}")

    print("\n  [C1] HR  (*** pre-confirmed significant):")
    p = mwu(agree["hr"], disagree["hr"], "I2 agrees E", "I2 disagrees E", "HR")
    results["C1_hr_agree_vs_disagree_p"] = round(p, 4)

    print("\n  [C2] Skin Temperature:")
    p = mwu(agree["temp"], disagree["temp"], "I2 agrees E", "I2 disagrees E", "Temp°C")
    results["C2_temp_agree_vs_disagree_p"] = round(p, 4)

    print("\n  [C3] EDA:")
    p = mwu(agree["eda"], disagree["eda"], "I2 agrees E", "I2 disagrees E", "EDA")
    results["C3_eda_agree_vs_disagree_p"] = round(p, 4)

    # ══════════════════════════════════════════════════════════════════════
    # TEST D — I2 vs I1: which is closer to E? (Hyp-7)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST D — Hyp-7: Does I2 track E better than I1? (Euclidean distance)\n")

    sub_d = phys.dropna(subset=["I2_val","I2_aro","i1_val","i1_aro","ev","ea"]).copy()
    sub_d["d_I2_E"] = sub_d.apply(lambda r: euc(r["I2_val"],r["I2_aro"],r["ev"],r["ea"]), axis=1)
    sub_d["d_I1_E"] = sub_d.apply(lambda r: euc(r["i1_val"],r["i1_aro"],r["ev"],r["ea"]), axis=1)
    sub_d["I2_closer"] = sub_d["d_I2_E"] < sub_d["d_I1_E"]

    n_total   = len(sub_d)
    n_i2_wins = int(sub_d["I2_closer"].sum())
    pct       = 100 * n_i2_wins / n_total if n_total else 0
    print(f"    I2 closer to E: {n_i2_wins}/{n_total}  ({pct:.1f}%)")

    diffs = sub_d["d_I1_E"] - sub_d["d_I2_E"]
    diffs_nz = diffs[diffs != 0]
    if len(diffs_nz) >= 5:
        stat_w, p_w = wilcoxon(diffs_nz, alternative="greater")
        sig = "*** p<0.05" if p_w < 0.05 else ("~   p<0.10" if p_w < 0.10 else "    n.s. ")
        print(f"    Wilcoxon (I2 closer): W={stat_w:.0f}  p={p_w:.4f}  [{sig}]")
        results["D_wilcoxon_I2_closer_p"] = round(p_w, 4)
    results["D_i2_wins_pct"] = round(pct, 2)

    print(f"\n    By E-quad:")
    for q in ["Q1","Q2","Q3","Q4"]:
        sq = sub_d[sub_d["E_quad"] == q]
        if len(sq) == 0: continue
        wins = sq["I2_closer"].sum()
        print(f"      {q}: {wins}/{len(sq)} ({100*wins/len(sq):.0f}%)  "
              f"d(I2,E)={sq['d_I2_E'].mean():.2f}  d(I1,E)={sq['d_I1_E'].mean():.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # TEST E — Switching (Drift) vs Non-Switching (Maintenance)  [Hyp-8]
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  TEST E — Hyp-8: Switching vs Non-Switching blocks")
    print("  Hypothesis: I1-I2 valence divergence is higher in Drift blocks;\n"
          "              Temp lower in Drift (active emotional shift = vasoconstriction)\n")

    sw  = phys[phys["block_type"].isin(["Drift_Block","Hyp9_Transition_Block"])].copy()
    nsw = phys[phys["block_type"].isin(["Maintenance_Block","Hyp9_Maintenance_Block"])].copy()
    print(f"    Switching n={len(sw)}, Non-switching n={len(nsw)}")

    # E1: |I1_val - I2_val|  (*** pre-confirmed p=0.035)
    sw["i1_i2_val_gap"]  = (sw["i1_val"]  - sw["I2_val"]).abs()
    nsw["i1_i2_val_gap"] = (nsw["i1_val"] - nsw["I2_val"]).abs()

    print("\n  [E1] |I1_val - I2_val|  (Hyp-8, *** pre-confirmed):")
    p = mwu(sw["i1_i2_val_gap"], nsw["i1_i2_val_gap"],
            "Switching (Drift)", "Non-switching (Maint)", "|I1-I2| val")
    results["E1_i1i2_val_sw_vs_nsw_p"] = round(p, 4)

    print("\n  [E2] Skin Temperature (switching vs non-switching):")
    p = mwu(sw["temp"], nsw["temp"], "Switching (Drift)", "Non-switching (Maint)", "Temp°C")
    results["E2_temp_sw_vs_nsw_p"] = round(p, 4)

    print("\n  [E3] HR:")
    p = mwu(sw["hr"], nsw["hr"], "Switching (Drift)", "Non-switching (Maint)", "HR")
    results["E3_hr_sw_vs_nsw_p"] = round(p, 4)

    print("\n  [E4] EDA:")
    p = mwu(sw["eda"], nsw["eda"], "Switching (Drift)", "Non-switching (Maint)", "EDA")
    results["E4_eda_sw_vs_nsw_p"] = round(p, 4)

    # ══════════════════════════════════════════════════════════════════════
    # BONUS: Temperature deep-dive — per E-quad
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{DASH}")
    print("  BONUS — Skin Temperature by E-quadrant (mean +/- std)\n")
    for q in ["Q1","Q2","Q3","Q4"]:
        sq = phys[(phys["E_quad"]==q) & phys["temp"].notna()]
        if len(sq) == 0: continue
        print(f"    {q}: n={len(sq):3d}  temp={sq['temp'].mean():.3f} +/- {sq['temp'].std():.3f}°C  "
              f"HR={sq['hr'].mean():.1f}  EDA={sq['eda'].mean():.3f}")

    # Q4 (calm positive) vs Q2 (negative high-arousal) — clearest contrast
    q4 = phys[(phys["E_quad"]=="Q4") & phys["temp"].notna()]
    q2 = phys[(phys["E_quad"]=="Q2") & phys["temp"].notna()]
    if len(q4) >= 3 and len(q2) >= 3:
        stat, p_q4q2 = mannwhitneyu(q4["temp"], q2["temp"], alternative="two-sided")
        sig = "*** p<0.05" if p_q4q2 < 0.05 else ("~   p<0.10" if p_q4q2 < 0.10 else "n.s.")
        print(f"\n    Q4 vs Q2 temp: U={stat:.0f}  p={p_q4q2:.4f}  [{sig}]")
        results["BONUS_temp_Q4_vs_Q2_p"] = round(p_q4q2, 4)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print("  FINAL SUMMARY")
    print(f"  {DASH}")
    tests = [
        ("A1", "Temp:  Negative vs Positive clips",          results.get("A1_temp_neg_vs_pos_p")),
        ("A2", "EDA:   Negative vs Positive clips",          results.get("A2_eda_neg_vs_pos_p")),
        ("A3", "HR:    Negative vs Positive clips",          results.get("A3_hr_neg_vs_pos_p")),
        ("B1", "Temp:  High-Arousal vs Low-Arousal",         results.get("B1_temp_hi_vs_lo_p")),
        ("B2", "HR:    High-Arousal vs Low-Arousal",         results.get("B2_hr_hi_vs_lo_p")),
        ("B3", "EDA:   High-Arousal vs Low-Arousal",         results.get("B3_eda_hi_vs_lo_p")),
        ("C1", "HR:    I2==E vs I2!=E  (confirmed)",         results.get("C1_hr_agree_vs_disagree_p")),
        ("C2", "Temp:  I2==E vs I2!=E  (confirmed)",         results.get("C2_temp_agree_vs_disagree_p")),
        ("C3", "EDA:   I2==E vs I2!=E  (confirmed)",         results.get("C3_eda_agree_vs_disagree_p")),
        ("D",  "Hyp-7: I2 closer to E than I1 (Wilcoxon)",  results.get("D_wilcoxon_I2_closer_p")),
        ("E1", "Hyp-8: |I1-I2| val — Drift vs Maint",       results.get("E1_i1i2_val_sw_vs_nsw_p")),
        ("E2", "Temp:  Drift vs Maintenance blocks",         results.get("E2_temp_sw_vs_nsw_p")),
        ("E3", "HR:    Drift vs Maintenance blocks",         results.get("E3_hr_sw_vs_nsw_p")),
        ("BONUS","Temp: Q4 (calm+pos) vs Q2 (neg+hi)",       results.get("BONUS_temp_Q4_vs_Q2_p")),
    ]
    for tid, desc, p in tests:
        if p is None or (isinstance(p, float) and math.isnan(p)):
            status = "  N/A    "
        elif p < 0.05:
            status = "*** PASS"
        elif p < 0.10:
            status = "~   MARG"
        else:
            status = "    n.s. "
        pstr = f"{p:.4f}" if (p == p and p is not None) else "nan"
        print(f"  [{tid:5s}] {status}  p={pstr:7s}  {desc}")
    print(SEP)

    # Save enriched data
    phys.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved enriched rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
